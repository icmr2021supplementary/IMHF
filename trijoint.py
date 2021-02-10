import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchwordemb
from args import get_parser
import numpy as np
from math import sqrt
import torch.nn.functional as F

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
max_length = 20

# # =============================================================================


def norm(input, p=2, dim=-1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class ingRNN(nn.Module):
    def __init__(self):
        super(ingRNN, self).__init__()
        word_dim = 300
        embed_size = 1024
        num_layers = 1
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=True)
        _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0)
        self.embs.weight.data.copy_(vec)

    def forward(self, x, sq_lengths):
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x)
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx \
            .view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(packed_seq)
        # Reshape *final* output to (batch_size, hidden_size)
        out, cap_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # bi-directional
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(out)
        out = out.gather(0, unsorted_idx).contiguous()

        out = (out[:, :, :out.size(2) // 2] + out[:, :, out.size(2) // 2:]) / 2
        I = torch.zeros(out.size(0), max_length - out.size(1), out.size(2)).cuda()
        if not len(I.size()) < 3:
            out = torch.cat((out, I), dim=1)

        return out


# Im2recipe model
class im2recipe(nn.Module):
    def __init__(self):
        super(im2recipe, self).__init__()
        if opts.preModel == 'resNet50':

            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.visionMLP = nn.Sequential(*modules)

            self.visual_embedding = nn.Sequential(
                nn.Linear(opts.imfeatDim, opts.embDim),
                nn.ReLU(inplace=True)
            )

        else:
            raise Exception('Only resNet50 model is implemented.')

        self.ingRNN_ = ingRNN()
        self.crosattn = CrossAttention(1024, 4, 0.0)
        self.crossgatefusion = GatedFusion(1024, 4, 0.0)

    def forward(self, x, y1, y2, z1, z2, keep="words", get_score=False):
        # recipe embedding
        ingr = self.ingRNN_(z1, z2)
        l_list = [int(i) for i in z2]
        ingr_mask = torch.ByteTensor([i * [1] + (max_length - i) * [0] for i in l_list]).cuda()

        instr = y1
        l_list = [int(i) for i in y2]
        instr_mask = torch.ByteTensor([i * [1] + (max_length - i) * [0] for i in l_list]).cuda()

        drive_num = torch.cuda.device_count()
        if keep == "words":
            ingr = ingr.unsqueeze(0).expand(drive_num, -1, -1, -1)
            cur_ingr_mask = ingr_mask.unsqueeze(0).expand(drive_num, -1, -1)
        ins_fuse, ingr_fuse = self.crosattn(instr, ingr, v1_mask=instr_mask, v2_mask=cur_ingr_mask)

        ingr_fuse = self.visual_embedding(ingr_fuse)
        ingr_fuse = norm(ingr_fuse)

        # visual embedding
        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), visual_emb.size(1), -1).transpose(1, 2)
        img = torch.mean(visual_emb, dim=1).squeeze()
        visual_emb = self.visual_embedding(visual_emb)
        visual_emb = norm(visual_emb)

        # score1
        score1 = torch.matmul(img, ins_fuse.t())
        # score2
        if keep == "words":
            ingr_fuse = ingr_fuse.unsqueeze(0).expand(drive_num, -1, -1, -1)
            cur_ingr_mask = ingr_mask.unsqueeze(0).expand(drive_num, -1, -1)

        scores2 = self.crossgatefusion(visual_emb, ingr_fuse, mask=cur_ingr_mask)

        scores = 0.2 * score1 + 0.8 * scores2

        if get_score:
            return scores

        # contrastive ranking loss
        diagonal = scores.diag().view(visual_emb.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (0.2 + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (0.2 + scores - d2).clamp(min=0)
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if True:  # max_violation
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def qkv_attention(query, key, value, q_mask=None, k_mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
    if k_mask is not None:
        scores.data.masked_fill_(k_mask.data.eq(0), -1e9)
    if q_mask is not None:
        scores.data.masked_fill_(q_mask.data.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        # h=8, d_model=1024
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears1 = nn.Linear(d_model, d_model)
        self.linears2 = nn.Linear(d_model, d_model)
        self.linears3 = nn.Linear(d_model, d_model)
        self.linears4 = nn.Linear(d_model, d_model)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        batches1 = query.size(0)
        batches2 = query.size(1)
        query = self.linears1(query).view(batches1, batches2, -1, self.h, self.d_k).transpose(2, 3)
        key = self.linears2(key).view(batches1, batches2, -1, self.h, self.d_k).transpose(2, 3)
        value = self.linears3(value).view(batches1, batches2, -1, self.h, self.d_k).transpose(2, 3)
        mask = mask.unsqueeze(-2).unsqueeze(-2)
        x, self.attn = qkv_attention(query, key, value, k_mask=mask)
        # "Concat" using a view and apply a final linear.
        x = x.transpose(2, 3).contiguous().view(batches1, batches2, -1, self.h * self.d_k)
        return self.linears4(x), self.attn


class CrossAttention(nn.Module):
    def __init__(self, dim, num_attn, dropout, reduce_func="mean"):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.h = num_attn
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.reduce_func = reduce_func

        self.img_key_fc = nn.Linear(dim, dim, bias=False)
        self.txt_key_fc = nn.Linear(dim, dim, bias=False)

        self.init_weights()
        print("CrossAttention module init success!")

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.dim +
                                  self.dim)
        self.img_key_fc.weight.data.uniform_(-r, r)
        self.txt_key_fc.weight.data.uniform_(-r, r)

    def forward(self, v1, v2, keep="words", v1_mask=None, v2_mask=None):
        if keep == "words":
            v2 = v2.squeeze(0)
            v2_mask = v2_mask.squeeze(0)
        elif keep == "regions":
            v1 = v1.squeeze(0)
            v1_mask = v1_mask.squeeze(0)

        k1 = self.img_key_fc(v1)
        k2 = self.txt_key_fc(v2)

        weighted_v1, attn_1 = qkv_attention(k2, k1, v1, k_mask=v1_mask.unsqueeze(-2),
                                            q_mask=v2_mask.unsqueeze(-1))
        weighted_v2, attn_2 = qkv_attention(k1, k2, v2, k_mask=v2_mask.unsqueeze(-2),
                                            q_mask=v1_mask.unsqueeze(-1))

        fused_v1 = torch.cat((v1, weighted_v2), dim=-1)
        fused_v2 = torch.cat((v2, weighted_v1), dim=-1)

        if self.reduce_func == "mean":
            fused_v1 = torch.mean(fused_v1, dim=-2)

        return fused_v1, fused_v2


def sum_attention(nnet, query, value, mask=None, dropout=None):
    scores = nnet(query).transpose(-2, -1)
    if mask is not None:
        scores.data.masked_fill_(mask.data.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SummaryAttn(nn.Module):
    def __init__(self, dim, num_attn, dropout, is_cat=False):
        super(SummaryAttn, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_attn),
        )
        self.h = num_attn
        self.is_cat = is_cat
        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, query, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-2)
        batch = query.size(0)

        weighted, self.attn = sum_attention(self.linear, query, value, mask=mask, dropout=self.dropout)
        weighted = weighted if self.is_cat else weighted.mean(dim=-2)

        return weighted


class GatedFusion(nn.Module):
    def __init__(self, dim, num_attn, dropout=0.01, reduce_func="mean", fusion_func="concat"):
        super(GatedFusion, self).__init__()
        self.dim = dim
        self.h = num_attn

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.reduce_func = reduce_func
        self.fusion_func = fusion_func

        self.img_key_fc = nn.Linear(dim, dim, bias=False)
        self.txt_key_fc = nn.Linear(dim, dim, bias=False)

        in_dim = dim
        if fusion_func == "sum":
            in_dim = dim
        elif fusion_func == "concat":
            in_dim = 2 * dim
        else:
            raise NotImplementedError('fusion error. Only support sum or concat.')

        self.fc_1 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), )
        self.fc_out = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        if reduce_func == "mean":
            self.reduce_layer = torch.mean
        elif reduce_func == "self_attn":
            self.reduce_layer_1 = SummaryAttn(dim, num_attn, dropout)
            self.reduce_layer_2 = SummaryAttn(dim, num_attn, dropout)

        self.mhattention = MultiHeadedAttention(8, 1024)

        self.init_weights()
        print("GatedFusion module init success!")

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.dim +
                                  self.dim)
        self.img_key_fc.weight.data.uniform_(-r, r)
        self.txt_key_fc.weight.data.uniform_(-r, r)
        self.fc_1[0].weight.data.uniform_(-r, r)
        self.fc_1[0].bias.data.fill_(0)
        self.fc_2[0].weight.data.uniform_(-r, r)
        self.fc_2[0].bias.data.fill_(0)
        self.fc_out[0].weight.data.uniform_(-r, r)
        self.fc_out[0].bias.data.fill_(0)
        self.fc_out[3].weight.data.uniform_(-r, r)
        self.fc_out[3].bias.data.fill_(0)

    def forward(self, v1, v2, get_score=True, keep="words", mask=None):
        if keep == "words":
            v2 = v2.squeeze(0)
            mask = mask.squeeze(0)
        elif keep == "regions":
            v1 = v1.squeeze(0)

        k1 = self.img_key_fc(v1)
        k2 = self.txt_key_fc(v2)
        batch_size_v1 = v1.size(0)
        batch_size_v2 = v2.size(0)

        v1 = v1.unsqueeze(1).expand(-1, batch_size_v2, -1, -1)
        k1 = k1.unsqueeze(1).expand(-1, batch_size_v2, -1, -1)
        v2 = v2.unsqueeze(0).expand(batch_size_v1, -1, -1, -1)
        k2 = k2.unsqueeze(0).expand(batch_size_v1, -1, -1, -1)
        mask = mask.unsqueeze(0).expand(batch_size_v1, -1, -1, -1)

        weighted_v1, attn_1 = self.mhattention(k2, k1, v1)
        if mask is not None:
            weighted_v2, attn_2 = self.mhattention(k1, k2, v2, mask=mask)
        else:
            weighted_v2, attn_2 = self.mhattention(k1, k2, v2)

        gate_v1 = torch.sigmoid((v1 * weighted_v2).sum(dim=-1)).unsqueeze(-1)
        gate_v2 = torch.sigmoid((v2 * weighted_v1).sum(dim=-1)).unsqueeze(-1)

        if self.fusion_func == "sum":
            fused_v1 = (v1 + weighted_v2) * gate_v1
            fused_v2 = (v2 + weighted_v1) * gate_v2
        elif self.fusion_func == "concat":
            fused_v1 = torch.cat((v1, weighted_v2), dim=-1) * gate_v1
            fused_v2 = torch.cat((v2, weighted_v1), dim=-1) * gate_v2

        co_v1 = self.fc_1(fused_v1) + v1
        co_v2 = self.fc_2(fused_v2) + v2

        if self.reduce_func == "self_attn":
            co_v1 = self.reduce_layer_1(co_v1, co_v1)
            co_v2 = self.reduce_layer_2(co_v2, co_v2, mask)
            co_v1 = norm(co_v1)
            co_v2 = norm(co_v2)
        else:
            co_v1 = self.reduce_layer(co_v1, dim=-2)
            co_v2 = self.reduce_layer(co_v2, dim=-2)
            co_v1 = norm(co_v1)
            co_v2 = norm(co_v2)

        if get_score:
            if self.fusion_func == "sum":
                score = self.fc_out(co_v1 + co_v2).squeeze(dim=-1)
            elif self.fusion_func == "concat":
                score = self.fc_out(torch.cat((co_v1, co_v2), dim=-1)).squeeze(dim=-1)

            if keep == "regions":
                score = score.transpose(0, 1)
            return score
        else:
            return torch.cat((co_v1, co_v2), dim=-1)