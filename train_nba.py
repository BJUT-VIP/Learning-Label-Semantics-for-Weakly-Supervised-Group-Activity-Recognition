import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.distributions import Categorical

import os

import copy
from tqdm import tqdm
import time
import random
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
# from clip.model import CLIP
# import clip
# from models.token_encoder_cross import CLIP

import models.models_nba as models
# import model.mm as mm
# from labelprediction.mplp import MPLP
# from mmcl import MMCL
from util.utils import *
from dataloader.dataloader import read_dataset

parser = argparse.ArgumentParser(description='Detector-Free Weakly Supervised Group Activity Recognition')
#/root/autodl-tmp/Datasets/Collective_activity_dataset/ActivityDataset/
# Dataset specification
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')#volleyball
parser.add_argument('--data_path', default='/data/yz/tm-detectorfree/dataset/NBA/', type=str, help='data path')#/data2/wq/videos/
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=18, type=int, help='number of frames for each clip')
parser.add_argument('--num_total_frame', default=72, type=int, help='number of total frames for each clip')
parser.add_argument('--num_activities', default=9, type=int, help='number of activity classes in volleyball dataset')

# Model parameters
parser.add_argument('--base_model', action='store_true', help='average pooling base model')
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')#inception_v3resnet18
parser.add_argument('--dilation', action='store_true', help='use dilation or not')#false/
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')
parser.add_argument('--num_classes', default=3493, type=int, help='transformer channel dimension')
parser.add_argument('--num_features', default=2048, type=int, help='transformer channel dimension')
parser.add_argument('--MPLPT', default=0.6, help='')
parser.add_argument('--DELTA', default=5, help='')
parser.add_argument('--R', default=0.01, help='')



# Motion parameters
parser.add_argument('--motion', default=False, action='store_true', help='use motion feature computation')
parser.add_argument('--multi_corr', default=False, help='motion correlation block at 4th and 5th')
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')

# Transformer parameters
parser.add_argument('--nheads', default=2, type=int, help='number of heads')
parser.add_argument('--enc_layers', default=1, type=int, help='number of encoder layers')#6
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_tokens', default=9, type=int, help='number of queries')

# Aggregation parameters
parser.add_argument('--nheads_agg', default=2, type=int, help='number of heads for partial context aggregation')

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=50, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=1, type=int, help='print frequency')
parser.add_argument('--batch', default=2, type=int, help='Batch size')
parser.add_argument('--test_batch', default=1, type=int, help='Test batch size')
parser.add_argument('--lr', default=5e-7, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=1e-5, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=1, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=20, type=int, help='step down size (cyclic) for learning rate scheduler')#40
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

# GPU
parser.add_argument('--device', default="1,2", type=str, help='GPU device')

# Load model
parser.add_argument('--load_model',default=True, help='load model')#action='store_true',True
parser.add_argument('--model_path', default="/data/yz/DF-NBA/result/nba-trainenhance-self-DD-2h2h1d-1e-3-again/epoch1_75.39%.pth", type=str, help='pretrained model path')
#/data/yz/DF-NBA/result/nba-trainenhance-self-DD-2h2h1d-1e-3-again/epoch1_75.39%.pth
#/data/yz/DF-NBA/result/nba-trainenhance-self-DD-2h2h1d-1e-3/epoch30_75.00%.pth
#"/data/yz/DF-NBA/result/nba-trainenhance-self-DD-2h2h1d-1e-3-ARG71.5/epoch1_75.78%.pth"
args = parser.parse_args()
best_mca = 0.0
best_mpca = 0.0
best_mca_epoch = 0
best_mpca_epoch = 0


def main():
    global args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # exp_name = '[%s]_DFGAR_<%s>' % (args.dataset, time_str)
    # exp_name ='VD-nomotion'
    exp_name = 'half_label_self_240.1'
    exp_name = 'label_cross_1hot'
    exp_name = '5label'
    exp_name = 'tra-6label'
    exp_name = 'debug1'
    exp_name = 'yuyi-6label'
    exp_name = 'trainenhance-6label'
    exp_name = 'trainenhance-6mutillabel'
    exp_name = 'trainenhance-6mutillabel+label'
    exp_name = 'trainenhance-6mutillabel-MLP'
    exp_name = 'trainenhance-6mutillabel-self-MLP'
    exp_name = 'trainenhance-6mutillabel-self-DD'
    exp_name = 'trainenhance-6mutillabel-self-ED'
    exp_name = 'trainenhance-6mutillabel-self-DD-2h4d-1e-3'
    exp_name = 'trainenhance-6mutillabel-self-DD-2h2d-1e-3'
    exp_name = 'trainenhance-6mutillabel-self-DD-2d8h-1e-3'
    exp_name = 'trainenhance-6mutillabel-self-DD-4d4h-1e-3'
    exp_name = 'trainenhance-6mutillabel-self-DD-3d4h-1e-3'
    exp_name = 'trainenhance-6mutillabel-self-DD1-2d4h-1e-3'
    exp_name = 'trainenhance-self-DD1-2d4h-1e-3'
    exp_name = 'trainenhance-self-2d4h-1e-3'
    exp_name = 'trainenhance-mybase-2d4h-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h4d-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h4d-1e-3-od'
    exp_name = 'nba-trainenhance-self-DD-1h4d-1e-3'
    exp_name = 'nba-trainenhance-self-DD-1h2d-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h1d-1e-3'
    exp_name = 'nba-trainenhance-self-DD-4h1d-1e-3-1'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-t21-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-t22-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-t44-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h2h2d-1e-3'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-1e-3-38'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-1e-3-again'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-1e-3-ARG1'
    exp_name = 'nba-trainenhance-self-DD-2h2h1d-1e-3-ARG1-again'

    save_path = './result/%s' % exp_name

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set = read_dataset(args)

    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=8, pin_memory=True)

    if args.base_model:
        # model = models.BaseModel(args)
        model = models.BaseModel_my(args)
        print("basemodel")
    else:
        model = models.DFGAR(args)
        # model = models.DFGAR_base(args)
        print("MY")
        # model = models. BaseModel_my(args)
        # model = models.BaseModel(args)
        # model = models.ADD_GCN(args)
        # model = models.BaseModel_weak(args)
        # model = models.ADD_GCN_label(args)

    model = torch.nn.DataParallel(model).cuda()

    # memory =1
    # text = text.to(device)
    # memory = mm.create('memory', args.num_features, args.num_classes)
    # memory = torch.nn.DataParallel(memory).cuda()
    # device = torch.device('cuda', 0)
    # text = text.to(device)
    # memory = memory.to(device)

    # model = torch.nn.parallel.DistributedDataParallel(model).cuda()

    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()###################################
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.max_lr, step_size_up=args.lr_step,
                                                  step_size_down=args.lr_step_down, mode='triangular2',
                                                  cycle_momentum=False)

    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = 1
    else:
        start_epoch = 1

    # training phase
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- %s at epoch #%d' % ("Train", epoch))
        # test_log = validate(test_loader, model, criterion, epoch)
        train_log = train(train_loader, model, criterion, optimizer, epoch)
        print_log(save_path, 'Accuracy: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                  (train_log['group_acc'], train_log['loss'], train_log['time']))
        print('Current learning rate is %f' % scheduler.get_last_lr()[0])
        scheduler.step()

        if epoch % args.test_freq == 0 :
            print_log(save_path, '----- %s at epoch #%d' % ("Test", epoch))
            test_log = validate(test_loader, model, criterion, epoch)
            print_log(save_path, 'Accuracy: %.2f%%, Mean-ACC: %.2f%%, Loss: %.4f, Using %.1f seconds' %
                      (test_log['group_acc'], test_log['mean_acc'], test_log['loss'], test_log['time']))

            print_log(save_path, '----------Best MCA: %.2f%% at epoch #%d.' %
                      (test_log['best_mca'], test_log['best_mca_epoch']))
            print_log(save_path, '----------Best MPCA: %.2f%% at epoch #%d.' %
                      (test_log['best_mpca'], test_log['best_mpca_epoch']))

            if epoch == test_log['best_mca_epoch'] or epoch == test_log['best_mpca_epoch']:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                result_path = save_path + '/epoch%d_%.2f%%.pth' % (epoch, test_log['group_acc'])
                torch.save(state, result_path)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    epoch_timer = Timer()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    for i, (_, images, activities, label_in, label0, label1, label2, label3) in tqdm(enumerate(train_loader)):#, label3
        images = images.cuda()                                      # [B, T, 3, H, W]################
        activities = activities.cuda()                              # [B, T]##################################
        label_in = label_in.cuda()
        label0 = label0.cuda()
        label1 = label1.cuda()
        label2 = label2.cuda()
        label3 = label3.cuda()
        label_3 = label2

        # tex = text(label_in)


        num_batch = images.shape[0]
        num_frame = images.shape[1]

        activities_in = activities[:, 0].reshape((num_batch, ))
        label_3in = label_3[:, 0].reshape((num_batch,))
        label0 = label0[:, 0].reshape((num_batch,))
        label1 = label1[:, 0].reshape((num_batch,))
        label2 = label2[:, 0].reshape((num_batch,))
        # label3 = label3[:, 0].reshape((num_batch,))
        # for j in range(len(num_batch)):
        #     if
        # label_in =


        # compute output
        scr, score = model(images)     # [B, C], scr_od

        criterion1 = torch.nn.MultiLabelSoftMarginLoss()
        loss1 = criterion1(scr, label_in)

        loss = loss1# + loss2# loss +
        # loss = criterion(score, activities_in)


        # measure accuracy and record loss
        # group_acc = accuracy(score, activities_in)
        # group_acc = accuracy1(scr, label0, label1, label2, label3)
        group_acc, pred1 = accuracy2(scr, label0, label1, label2)#, scr_od, label3,activities_in
        losses.update(loss, num_batch)
        accuracies.update(group_acc, num_batch)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': accuracies.avg * 100.0,
    }

    return train_log


@torch.no_grad()
def validate(test_loader, model, criterion, epoch):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    epoch_timer = Timer()
    losses = AverageMeter()
    accuracies = AverageMeter()
    true = []
    pred = []

    # switch to eval mode
    model.eval()



    # labelpred = MPLP(args.MPLPT)
    # criterion1 = MMCL(args.DELTA, args.R).to(device)

    for i, (_, images, activities, label_in, label0, label1, label2, label3) in tqdm(enumerate(test_loader)):#, label3
        images = images.cuda()##################################
        activities = activities.cuda()###################################
        label_in = label_in.cuda()
        label0 = label0.cuda()
        label1 = label1.cuda()
        label2 = label2.cuda()
        label3 = label3.cuda()
        label_3 = label2

        num_batch = images.shape[0]
        num_frame = images.shape[1]
        activities_in = activities[:, 0].reshape((num_batch,))
        label_3in = label_3[:, 0].reshape((num_batch,))
        label0 = label0[:, 0].reshape((num_batch,))
        label1 = label1[:, 0].reshape((num_batch,))
        label2 = label2[:, 0].reshape((num_batch,))
        # label3 = label3[:, 0].reshape((num_batch,))



        # compute output
        # score,_ = model(images)
        scr, score = model(images)  # [B, C], scr_od

        criterion1 = torch.nn.MultiLabelSoftMarginLoss()
        loss = criterion1(scr, label_in)

        loss = loss #+ loss2# loss +

        true = true + activities_in.tolist()
        # pred = pred + torch.argmax(score, dim=1).tolist()


        loss = criterion(score, activities_in)

        # measure accuracy and record loss
        # group_acc = accuracy(score, activities_in)
        # group_acc = accuracy1(scr, label0, label1, label2, label3)
        # group_acc = accuracy2(scr, label0, label1, label2)
        group_acc, pred1 = accuracy2(scr, label0, label1, label2)#, scr_od, label3, activities_in
        pred = pred + pred1#.tolist()
        losses.update(loss, num_batch)
        accuracies.update(group_acc, num_batch)

    acc = accuracies.avg * 100.0
    confusion = confusion_matrix(true, pred)
    mean_acc = np.mean([confusion[i, i] / confusion[i, :].sum() for i in range(confusion.shape[0])]) * 100.0

    if acc > best_mca:
        best_mca = acc
        best_mca_epoch = epoch
    if mean_acc > best_mpca:
        best_mpca = mean_acc
        best_mpca_epoch = epoch

    test_log = {
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': acc,
        'mean_acc': mean_acc,
        'best_mca': best_mca,
        'best_mpca': best_mpca,
        'best_mca_epoch': best_mca_epoch,
        'best_mpca_epoch': best_mpca_epoch,
    }

    return test_log


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


class Timer(object):
    """
    class to do timekeeping
    """
    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


def accuracy(output, target):
    output = torch.argmax(output, dim=1)
    tt = torch.eq(target.int(), output.int())
    correct = torch.sum(torch.eq(target.int(), output.int())).float()
    return correct.item() / output.shape[0]

def accuracy1(scr, label0, label1, label2, label3):

    output0 = torch.argmax(scr[:, :2], dim=1)
    output1 = torch.argmax(scr[:, 2:4], dim=1)
    output2 = torch.argmax(scr[:, 4:6], dim=1)
    output3 = torch.argmax(scr[:, 6:], dim=1)
    tt0 = torch.eq(label0.int(), output0.int())
    tt1 = torch.eq(label1.int(), output1.int())
    tt2 = torch.eq(label2.int(), output2.int())
    tt3 = torch.eq(label3.int(), output3.int())
    tt = tt0.int() * tt1.int() * tt2.int() * tt3.int()
    correct = torch.sum(tt).float()

    # correct = torch.sum(torch.eq(target.int(), output.int())).float()
    # return correct.item() / output.shape[0]
    return correct.item() / scr.shape[0]

# def accuracy2(scr, label0, label1, label2):
#
#     output0 = torch.argmax(scr[:, :2], dim=1)
#     output1 = torch.argmax(scr[:, 2:4], dim=1)
#     output2 = torch.argmax(scr[:, 4:], dim=1)
#     # output3 = torch.argmax(scr[:, 6:], dim=1)
#     tt0 = torch.eq(label0.int(), output0.int())
#     tt1 = torch.eq(label1.int(), output1.int())
#     tt2 = torch.eq(label2.int(), output2.int())
#     # tt3 = torch.eq(label3.int(), output3.int())
#     tt = tt0.int() * tt1.int() * tt2.int()
#     correct = torch.sum(tt).float()
#
#     # correct = torch.sum(torch.eq(target.int(), output.int())).float()
#     # return correct.item() / output.shape[0]
#     return correct.item() / scr.shape[0]

def accuracy2(scr, label0, label1, label2):
    pred1 = []
    output0 = torch.argmax(scr[:, :2], dim=1)
    output1 = torch.argmax(scr[:, 2:4], dim=1)
    output2 = torch.argmax(scr[:, 4:], dim=1)
    if output1[0] == 0:
        if output2[0] ==0:
            pred1 = [3]
        elif output2[0] ==2:
            pred1 = [4]
        elif output2[0] ==1:
            pred1 = [5]
    if output1[0] == 1:
        if output0[0] == 0 and output2[0] ==0:
            pred1 = [0]
        elif output0[0] == 0 and output2[0] ==2:
            pred1 = [1]
        elif output0[0] == 0 and output2[0] ==1:
            pred1 = [2]
    # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==0:
    #     pred1 = [3]
    # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==2:
    #     pred1 = [4]
    # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==1:
    #     pred1 = [5]
        elif output0[0] == 1 and output2[0] ==0:
            pred1 = [6]
        elif output0[0] == 1 and output2[0] ==2:
            pred1 = [7]
        elif output0[0] == 1 and output2[0] ==1:
            pred1 = [8]
        else:
            pred1 = [8]
            # print(label0, label1, label2)
            # print(output0[0],output1[0],output2[0])
        # print(label0, label1, label2)
    # output3 = torch.argmax(scr[:, 6:], dim=1)
    tt0 = torch.eq(label0.int(), output0.int())
    tt1 = torch.eq(label1.int(), output1.int())
    tt2 = torch.eq(label2.int(), output2.int())
    # tt3 = torch.eq(label3.int(), output3.int())
    tt = tt0.int() * tt1.int() * tt2.int()
    correct = torch.sum(tt).float()

    # correct = torch.sum(torch.eq(target.int(), output.int())).float()
    # return correct.item() / output.shape[0]
    return correct.item() / scr.shape[0], pred1

# def accuracy2(scr, label0, label1, label2, scr_od, label3,activities_in):
#
#     output0 = torch.argmax(scr[:, :2], dim=1)
#     output1 = torch.argmax(scr[:, 2:4], dim=1)
#     scr3 = F.softmax(scr[:, 4:],dim=1)
#     scr_od = F.softmax(scr_od[:,:],dim=1)
#     output2 = torch.argmax(scr_od[:,:], dim=1)
#     # if output1[0] == 0:
#     #     if output2[0] ==0:
#     #         pred1 = [3]
#     #     elif output2[0] ==2:
#     #         pred1 = [4]
#     #     elif output2[0] ==1:
#     #         pred1 = [5]
#     # if output1[0] == 1:
#     #     if output0[0] == 0 and output2[0] ==0:
#     #         pred1 = [0]
#     #     elif output0[0] == 0 and output2[0] ==2:
#     #         pred1 = [1]
#     #     elif output0[0] == 0 and output2[0] ==1:
#     #         pred1 = [2]
#     # # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==0:
#     # #     pred1 = [3]
#     # # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==2:
#     # #     pred1 = [4]
#     # # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==1:
#     # #     pred1 = [5]
#     #     elif output0[0] == 1 and output2[0] ==0:
#     #         pred1 = [6]
#     #     elif output0[0] == 1 and output2[0] ==2:
#     #         pred1 = [7]
#     #     elif output0[0] == 1 and output2[0] ==1:
#     #         pred1 = [8]
#
#     tt0 = torch.eq(label0.int(), output0.int())
#     tt1 = torch.eq(label1.int(), output1.int())
#     tt2 = torch.eq(label2.int(), output2.int())
#
#
#     tt = tt0.int() * tt1.int() * tt2.int()
#     # tt = torch.eq(activities_in.int(), pred1.int())
#     correct = torch.sum(tt).float()
#
#     # correct = torch.sum(torch.eq(target.int(), output.int())).float()
#     # return correct.item() / output.shape[0]
#     return correct.item() / scr.shape[0]

def read_labels(path):
    """
    reading annotations for the given sequence
    """
    annotations = []
    annotations = [[1, 0, 1, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0],
                   [1, 0, 0, 0, 1, 0],
                   [1, 0, 0, 0, 0, 1],
                   [0, 1, 1, 0, 0, 0],
                   [0, 1, 0, 1, 0, 0],
                   [0, 1, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0, 1]
                   ]

    # annotations = [[1, 0, 0, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 0, 0, 0, 0],
    #                [0, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 0, 0, 1, 0, 0, 0, 0],
    #                [0, 0, 0, 0, 1, 0, 0, 0],
    #                [0, 0, 0, 0, 0, 1, 0, 0],
    #                [0, 0, 0, 0, 0, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 0, 1]
    #                ]



    # with open(path) as f:
    #     for l in f.readlines():
    #         values = l[:-1].split('\0')
    #         # label = values[0]
    #         # values = values[2:]
    #         annotations = annotations + values

    return annotations

if __name__ == '__main__':
    main()
