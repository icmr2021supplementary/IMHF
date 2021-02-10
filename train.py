import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader 
from args import get_parser
from trijoint import im2recipe
import pickle

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():

    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.crosattn = torch.nn.DataParallel(model.crosattn)
    model.crossgatefusion = torch.nn.DataParallel(model.crossgatefusion)
    model.cuda()

    # creating different parameter groups
    vision_params = list(map(id, model.visionMLP.parameters()))
    base_params   = filter(lambda p: id(p) not in vision_params, model.parameters())
   
    # optimizer - with lr initialized accordingly
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.visionMLP.parameters(), 'lr': opts.lr*opts.freeVision }
            ], lr=opts.lr*opts.freeRecipe)

    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val = float('inf')
    else:
        best_val = float('inf')

    valtrack = 0

    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial vision params lr: %f' % optimizer.param_groups[1]['lr'])

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    cudnn.benchmark = True

    # preparing the training laoder
    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
            transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(256), # we get only the center of that rescaled
            transforms.RandomCrop(224), # random crop within the center crop 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),data_path=opts.data_path,partition='train'),
        batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.workers, pin_memory=True)
    print('Training loader prepared.')

    # preparing validation loader
    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
            transforms.Compose([
            transforms.Scale(256), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(224), # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize,
        ]),data_path=opts.data_path,partition='val'),
        batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.workers, pin_memory=True)
    print('Validation loader prepared.')

    # run epochs
    for epoch in range(opts.start_epoch, opts.epochs):

        # train
        loss_avg = train(train_loader, model, optimizer, epoch)

        # validation
        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            val_loss = validate(val_loader, model)
            # check patience
            if val_loss >= best_val:
                valtrack += 1
            else:
                valtrack = 0
            if valtrack >= opts.patience:
                # switch modalities
                opts.freeVision = opts.freeRecipe;
                opts.freeRecipe = not (opts.freeVision)
                # change lr accordingly
                adjust_learning_rate(optimizer, epoch, opts)
                valtrack = 0

            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': valtrack,
                'freeVision': opts.freeVision,
                'curr_val': val_loss,
            }, is_best)

            print("Validation, epoch:", epoch, "best_val:", best_val, "valtrack:", valtrack)


def train(train_loader, model, optimizer, epoch):

    losses_record = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input_var = list() 
        for j in range(len(input)):
            input_var.append(input[j].cuda())

        target_var = list()
        for j in range(len(target)):
            target_var.append(target[j].cuda())

        # compute output
        loss = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print("item:", i, "loss:", loss.data)

        losses_record.update(loss.data, input[0].size(0))

    return losses_record.avg


def validate(val_loader, model):

    # switch to evaluate mode
    model.eval()

    # we save the input features for the first validation epoch.
    # After the first validation, we read from files to save time.
    if opts.val_first:
        filenum = 1
        f1 = open(opts.path_results + 'img_embeds.pkl', 'wb')
        f2 = open(opts.path_results + 'rec_embeds1.pkl', 'wb')
        f3 = open(opts.path_results + 'rec_embeds2.pkl', 'wb')
        f4 = open(opts.path_results + 'rec_embeds3.pkl', 'wb')
        f5 = open(opts.path_results + 'rec_embeds4.pkl', 'wb')
        f6 = open(opts.path_results + 'img_ids.pkl', 'wb')
        f7 = open(opts.path_results + 'rec_ids.pkl', 'wb')
        for i, (input, target) in enumerate(val_loader):
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
                print("save:", filenum, "done.")
                if filenum == 10:
                    f1.close()
                    f2.close()
                    f3.close()
                    f4.close()
                    f5.close()
                    f6.close()
                    f7.close()
                    print("Save 10*1000 done.")
                    break
                filenum = filenum + 1
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
        opts.val_first = False

    N = 1000     # opts.medr
    type_embedding = opts.embtype
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
        for i in range(10):  # 10
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
                    score_sub = model(torch.from_numpy(im_sub[p * 50:(p + 1) * 50]).cuda(),
                                      torch.from_numpy(rec1_sub[q * 50:(q + 1) * 50]).cuda(),
                                      torch.from_numpy(rec2_sub[q * 50:(q + 1) * 50]).cuda(),
                                      torch.from_numpy(rec3_sub[q * 50:(q + 1) * 50]).cuda(),
                                      torch.from_numpy(rec4_sub[q * 50:(q + 1) * 50]).cuda(), get_score=True)
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

            for i in recall.keys():
                glob_recall[i] += recall[i]
            glob_rank.append(med)

        for i in glob_recall.keys():
            glob_recall[i] = glob_recall[i] / 10
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()
        f7.close()
        print("medR:", np.average(glob_rank))
        print("recall:", glob_recall)
    return np.average(glob_rank)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    filename = opts.snapshots + 'model_e%03d_v-%.3f.pth.tar' % (state['epoch'],state['best_val'])
    if is_best:
        torch.save(state, filename)


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


def adjust_learning_rate(optimizer, epoch, opts):
    """Switching between modalities"""
    if (epoch + 1) % 25 == 0 and epoch != 0:
        opts.lr = opts.lr * 0.1
    # parameters corresponding to the rest of the network
    optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
    # parameters corresponding to visionMLP 
    optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision

    print("epoch:", epoch, "change branch.")
    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial vision lr: %f' % optimizer.param_groups[1]['lr'])

    # after first modality change we set patience to 4
    opts.patience = 4


if __name__ == '__main__':
    main()
