import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader
import numpy as np
from trijoint import im2recipe
import pickle
from args import get_parser
import random
# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(opts.seed)
np.random.seed(opts.seed)


def main():
    saved = True  # True
    if saved:
        model = im2recipe()
        model.visionMLP = torch.nn.DataParallel(model.visionMLP)
        model.crosattn = torch.nn.DataParallel(model.crosattn)
        model.crossgatefusion = torch.nn.DataParallel(model.crossgatefusion)
        model.cuda()

        print("=> loading checkpoint '{}'".format(opts.model_path))
        checkpoint = torch.load(opts.model_path, encoding='latin1')
        opts.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opts.model_path, checkpoint['epoch']))

        # run test
        model.eval()
        random.seed(opts.seed)
        # Ranker
        N = 1000  # opts.medr
        type_embedding = 'image'   # image
        idxs = range(N)

        glob_rank = []
        glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
        f1 = open(opts.path_results + 'img_embeds.pkl', 'rb')
        f2 = open(opts.path_results + 'rec_embeds1.pkl', 'rb')
        f3 = open(opts.path_results + 'rec_embeds2.pkl', 'rb')
        f4 = open(opts.path_results + 'rec_embeds3.pkl', 'rb')
        f5 = open(opts.path_results + 'rec_embeds4.pkl', 'rb')
        f6 = open(opts.path_results + 'img_ids.pkl', 'rb')
        f7 = open(opts.path_results + 'rec_ids.pkl', 'rb')
        with torch.no_grad():
            for i in range(10):
                img = pickle.load(f1)
                rec1 = pickle.load(f2)
                rec2 = pickle.load(f3)
                rec3 = pickle.load(f4)
                rec4 = pickle.load(f5)
                img_ids = pickle.load(f6)
                rec_ids = pickle.load(f7)

                names = rec_ids
                ids = random.sample(range(0, len(names)), N)
                im_sub = img[ids]
                rec1_sub = rec1[ids]
                rec2_sub = rec2[ids]
                rec3_sub = rec3[ids]
                rec4_sub = rec4[ids]
                ids_sub = names[ids]
                ids_sub_img = img_ids[ids]

                score_list = []
                for p in range(20):
                    score_sub_list = []
                    for q in range(20):
                        score_sub = model(torch.from_numpy(im_sub[p*50:(p+1)*50]).cuda(), torch.from_numpy(rec1_sub[q*50:(q+1)*50]).cuda(), torch.from_numpy(rec2_sub[q*50:(q+1)*50]).cuda(), torch.from_numpy(rec3_sub[q*50:(q+1)*50]).cuda(), torch.from_numpy(rec4_sub[q*50:(q+1)*50]).cuda(), get_score=True)
                        score_sub_list.append(score_sub)
                    score = torch.cat(score_sub_list, dim=1)
                    score_list.append(score)
                scores = torch.cat(score_list, dim=0)
                if type_embedding == 'image':
                    sims = scores.data.cpu().numpy()  # for im2recipe
                else:
                    sims = scores.data.cpu().numpy().T  # for recipe2im

                med_rank = []
                recall = {1: 0.0, 5: 0.0, 10: 0.0}
                for ii in idxs:

                    name = ids_sub[ii]
                    # get a column of similarities
                    sim = sims[ii, :]

                    # sort indices in descending order
                    sorting = np.argsort(sim)[::-1].tolist()

                    # find where the index of the pair sample ended up in the sorting
                    pos = sorting.index(ii)

                    if (pos + 1) == 1:
                        recall[1] += 1
                    if (pos + 1) <= 5:
                        recall[5] += 1
                    if (pos + 1) <= 10:
                        recall[10] += 1

                    # store the position
                    med_rank.append(pos + 1)

                for i in recall.keys():
                    recall[i] = recall[i] / N

                med = np.median(med_rank)
                # print "median", med

                for i in recall.keys():
                    glob_recall[i] += recall[i]
                glob_rank.append(med)

            for i in glob_recall.keys():
                glob_recall[i] = glob_recall[i] / 10    # 10

            print("medR:", np.average(glob_rank))
            print("recall:", glob_recall)
            f1.close()
            f2.close()
            f3.close()
            f4.close()
            f5.close()
            f6.close()
            f7.close()
    else:
        # data preparation, loaders
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # preparing test loader
        test_loader = torch.utils.data.DataLoader(
            ImagerLoader(opts.img_path,
                         transforms.Compose([
                             transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                             transforms.CenterCrop(224),  # we get only the center of that rescaled
                             transforms.ToTensor(),
                             normalize,
                         ]), data_path=opts.data_path, partition='test'),
            batch_size=opts.batch_size, shuffle=True,
            num_workers=opts.workers, pin_memory=True)
        print('Test loader prepared.')
        num = 1
        f1 = open(opts.path_results + 'img_embeds.pkl', 'wb')
        f2 = open(opts.path_results + 'rec_embeds1.pkl', 'wb')
        f3 = open(opts.path_results + 'rec_embeds2.pkl', 'wb')
        f4 = open(opts.path_results + 'rec_embeds3.pkl', 'wb')
        f5 = open(opts.path_results + 'rec_embeds4.pkl', 'wb')
        f6 = open(opts.path_results + 'img_ids.pkl', 'wb')
        f7 = open(opts.path_results + 'rec_ids.pkl', 'wb')
        for i, (input, target) in enumerate(test_loader):
            input_var = list()
            for j in range(len(input)):
                input_var.append(input[j])
            target_var = list()
            for j in range(len(target)):
                target_var.append(target[j])

            if i == 0:
                data0 = input_var[0].data.numpy()
                data1 = input_var[1].data.numpy()
                data2 = input_var[2].data.numpy()
                data3 = input_var[3].data.numpy()
                data4 = input_var[4].data.numpy()
                imgid = target[-2]
                recid = target[-1]
            elif len(data0) > 1000:
                pickle.dump(data0, f1)
                pickle.dump(data1, f2)
                pickle.dump(data2, f3)
                pickle.dump(data3, f4)
                pickle.dump(data4, f5)
                pickle.dump(imgid, f6)
                pickle.dump(recid, f7)
                print("save:", num, "done.")
                if num == 10:
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    print("Save 10*1000 done.")
                    break
                num = num + 1
                data0 = input_var[0].data.numpy()
                data1 = input_var[1].data.numpy()
                data2 = input_var[2].data.numpy()
                data3 = input_var[3].data.numpy()
                data4 = input_var[4].data.numpy()
                imgid = target[-2]
                recid = target[-1]
            else:
                data0 = np.concatenate((data0, input_var[0].data.numpy()), axis=0)
                data1 = np.concatenate((data1, input_var[1].data.numpy()), axis=0)
                data2 = np.concatenate((data2, input_var[2].data.numpy()), axis=0)
                data3 = np.concatenate((data3, input_var[3].data.numpy()), axis=0)
                data4 = np.concatenate((data4, input_var[4].data.numpy()), axis=0)
                imgid = np.concatenate((imgid, target[-2]), axis=0)
                recid = np.concatenate((recid, target[-1]), axis=0)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
