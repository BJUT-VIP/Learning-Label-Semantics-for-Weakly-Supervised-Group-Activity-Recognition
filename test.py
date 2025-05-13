import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.distributions import Categorical

import os
import copy
import time
import random
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import models.models as models
from util.utils import *
from dataloader.dataloader import read_dataset

import matplotlib.pyplot as plt
# import pylab
# import matplotlib.image as mpimg
from PIL import Image
import PIL
import torchvision
from torchvision.transforms import functional
import itertools


parser = argparse.ArgumentParser(description='Detector-Free Weakly Supervised Group Activity Recognition')

# Dataset specification
parser.add_argument('--dataset', default='volleyball', type=str, help='volleyball or nba')
# parser.add_argument('--data_path', default='/data2/wq/videos/', type=str, help='data path')
parser.add_argument('--data_path', default='/dataset/', type=str, help='data path')
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=5, type=int, help='number of frames for each clip')
parser.add_argument('--num_total_frame', default=10, type=int, help='number of total frames for each clip')
parser.add_argument('--num_activities', default=8, type=int, help='number of activity classes in volleyball dataset')

# Model parameters
parser.add_argument('--base_model', action='store_true', help='average pooling base model')
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')
parser.add_argument('--dilation', action='store_true', help='use dilation or not')
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

# Motion parameters
parser.add_argument('--motion', default=False, action='store_true', help='use motion feature computation')
parser.add_argument('--multi_corr', action='store_true', help='motion correlation block at 4th and 5th')
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')

# Transformer parameters
parser.add_argument('--nheads', default=4, type=int, help='number of heads')    # 4
parser.add_argument('--enc_layers', default=2, type=int, help='number of encoder layers')   # 2
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_tokens', default=8, type=int, help='number of queries')

# Aggregation parameters
parser.add_argument('--nheads_agg', default=4, type=int, help='number of heads for partial context aggregation')

# Training parameters
parser.add_argument('--batch', default=1, type=int, help='Batch size')
parser.add_argument('--test_batch', default=1, type=int, help='Test batch size')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')

parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=200, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=2, type=int, help='print frequency')
# parser.add_argument('--batch', default=4, type=int, help='Batch size')
# parser.add_argument('--test_batch', default=4, type=int, help='Test batch size')
parser.add_argument('--lr', default=1e-6, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=1e-4, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
# parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

# GPU
parser.add_argument('--device', default="0", type=str, help='GPU device')

# Load model   /root/autodl-tmp/Projects/tm/df/result/trainenhance-6mutillabel-self-DD-2d4h-1e-3/epoch24_92.37%.pth
# parser.add_argument('--model_path', default="/data/yz/DF-NBA/trainenhance-self-DD-2h4d-1e-3-again/epoch44_92.45%25.pth", type=str, help='pretrained model path')
# parser.add_argument('--model_path', default="/data/frg/code/tmm/trainenhance-self-DD-2h4d-1e-3-again/epoch44_92.45.pth", type=str, help='pretrained model path')
parser.add_argument('--model_path', default="/data/frg/code/tmm/result/vd/epoch28_92.45%.pth", type=str, help='pretrained model path')

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
    exp_name = 'debug1'
    save_path = './result/%s' % exp_name
    name = '/root/autodl-tmp/Projects/tm/DF/result/debug/2'

    _, test_set = read_dataset(args)

    test_loader = data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=8, pin_memory=True)

    if args.base_model:
        # model = models.BaseModel(args)
        model = models.BaseModel_my(args)
        print("basemodel")
    else:
        # model = models.DFGAR(args)
        # model = models.BaseModel_my(args)
        # model = models.ADD_GCN(args)
        model = models.DFGAR(args)
        print("MY")
    model = torch.nn.DataParallel(model).cuda()

    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # define loss function and optimizer
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # 2024.5.29  查看模型参数
    # content = torch.load('t.pth')
    # print(checkpoint.keys())
    # # print(checkpoint['epoch'])
    # # print(checkpoint['state_dict'])
    # print(checkpoint['optimizer'])
    # print(checkpoint['scheduler'])

    # validate(train_loader, model)
    acc, mean_acc, confusion = validate(test_loader, model)
    print('Accuracy is %.2f' % acc)
    print('Mean accuracy is %.2f' % mean_acc)
    print(confusion)
    # visualize(confusion, '/data/yz/DF-NBA/trainenhance-self-DD-2h4d-1e-3-again/VD92.5')
    visualize(confusion, '/data/frg/code/tmm/result/vd/VD92.5')


@torch.no_grad()
def validate(test_loader, model):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    accuracies = AverageMeter()
    true = []
    pred = []

    # switch to eval mode
    model.eval()

    for i, (ls, images, activities, label_in, label0, label1) in tqdm(enumerate(test_loader)):
        images = images.cuda()
        activities = activities.cuda()
        label_in = label_in.cuda()
        label0 = label0.cuda()
        label1 = label1.cuda()

        num_batch = images.shape[0]
        num_frame = images.shape[1]
        activities_in = activities[:, 0].reshape((num_batch,))
        label0 = label0[:, 0].reshape((num_batch,))
        label1 = label1[:, 0].reshape((num_batch,))

        # compute output
        scr, score = model(images)

        criterion1 = torch.nn.MultiLabelSoftMarginLoss()
        loss1 = criterion1(scr, label_in)

        # images = images.mean(dim=0)[2, :, :, :]
        # image = PIL.Image.fromarray(torch.clamp(images * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
        # image = torchvision.transforms.functional.to_pil_image(images)
        #
        # #visual
        # output = output.transpose(0, 1).mean(dim=0)
        # output = output.cpu().numpy()[2, :, :]#.squeeze(0)
        # output = output
        #
        # tang_max_stage1 = output[:, :].max()
        # tang_min_stage1 = output[:, :].min()
        # tang_array_stage1 = (output[:, :] - tang_min_stage1) / (
        #         tang_max_stage1 - tang_min_stage1)
        #
        # # output = tang_array_stage1
        # output = Image.fromarray(350*tang_array_stage1)
        # if output.mode == "F":
        #     output = output.convert('RGB')
        #
        # # output.save('/data/yz/ARG_6/vision/eval_net_ijcai/results/' + str(sid) + '_' + str(src_fid) +  '.jpg')
        # # print(str(sid) +str(src_fid))
        # [sid, src_fid, fid] = ls[2]
        # sid = sid.numpy()
        # src_fid = src_fid.numpy()
        # fid = fid.numpy()
        #
        # dataset_path = "/root/autodl-tmp/Datasets/Volleyball_dataset/videos/"
        # img_path = dataset_path + '/' + str(sid[0]) + '/' + str(src_fid[0]) + '/' + str(fid[0]) + '.jpg'
        # img = Image.open(img_path)
        #
        #
        # output = output.resize((1280, 720))
        # plt.imshow(img)
        # plt.imshow(output, cmap=plt.get_cmap('jet'), alpha=0.5)
        #
        # plt.axis("off")
        #
        # save_img_path_2 = '/root/autodl-tmp/Projects/tm/DF/result/2/' + str(i) + '.jpg'
        # plt.savefig(save_img_path_2, bbox_inches='tight', dpi=600, pad_inches=0.0)
        # plt.close("all")

        true = true + activities_in.tolist()
        # pred = pred + torch.argmax(score, dim=1).tolist()

        # measure accuracy and record loss
        group_acc, pred1 = accuracy1(scr, label0, label1)
        pred = pred + pred1
        # group_acc = accuracy(score, activities_in)
        accuracies.update(group_acc, num_batch)

    acc = accuracies.avg * 100.0
    # confusion = ConfusionMatrix.update(true, pred)
    confusion = confusion_matrix(true, pred)
    mean_acc = np.mean([confusion[i, i] / confusion[i, :].sum() for i in range(confusion.shape[0])]) * 100.0

    return acc, mean_acc, confusion


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
    correct = torch.sum(torch.eq(target.int(), output.int())).float()
    return correct.item() / output.shape[0]

def accuracy1(scr, label0, label1):
    pred1 = []
    output0 = torch.argmax(scr[:, :2], dim=1)
    output1 = torch.argmax(scr[:, 2:], dim=1)
    if output0[0] == 0 and output1[0] ==0:
        pred1 = [0]
    elif output0[0] == 0 and output1[0] == 1:
        pred1 = [1]
    elif output0[0] == 0 and output1[0] == 2:
        pred1 = [2]
    elif output0[0] == 0 and output1[0] == 3:
        pred1 = [3]
    elif output0[0] == 1 and output1[0] == 0:
        pred1 = [4]
    elif output0[0] == 1 and output1[0] == 1:
        pred1 = [5]
    elif output0[0] == 1 and output1[0] == 2:
        pred1 = [6]
    elif output0[0] == 1 and output1[0] == 3:
        pred1 = [7]
    tt0 = torch.eq(label0.int(), output0.int())
    tt1 = torch.eq(label1.int(), output1.int())
    tt = tt0.int() * tt1.int()
    correct = torch.sum(tt).float()

    # correct = torch.sum(torch.eq(target.int(), output.int())).float()
    # return correct.item() / output.shape[0]
    return correct.item() / scr.shape[0], pred1

def visualize(matrix, name):
    num_class = 8
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print(cm)
    plt.figure()  # TITAN可去掉AGG
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(num_class)
    plt.xticks(tick_marks, ('r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass', 'l_winpoint'))
    plt.yticks(tick_marks, ('r_set', 'r_spike', 'r-pass', 'r_winpoint', 'l_set', 'l-spike', 'l-pass', 'l_winpoint'))

    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.savefig(name + "_confusemat.png")





if __name__ == '__main__':
    main()
