import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.distributions import Categorical
import cv2
from skimage import io
import skimage
import os
import copy
import time
import random
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# import models.models_nba as models
import models.models_nba_now as models
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
#/root/autodl-tmp/Datasets/Collective_activity_dataset/ActivityDataset/
# Dataset specification
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')#volleyball
parser.add_argument('--data_path', default='/dataset/', type=str, help='data path')#/data2/wq/videos/
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
parser.add_argument('--ratio', default=1, type=float, help='')



# Motion parameters
parser.add_argument('--motion', default=False, action='store_true', help='use motion feature computation')
parser.add_argument('--multi_corr', default=False, help='motion correlation block at 4th and 5th')
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')

# Transformer parameters
parser.add_argument('--nheads', default=2, type=int, help='number of heads')
parser.add_argument('--enc_layers', default=2, type=int, help='number of encoder layers')#6
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_tokens', default=9, type=int, help='number of queries')

# Aggregation parameters
parser.add_argument('--nheads_agg', default=2, type=int, help='number of heads for partial context aggregation')

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')  # 1
parser.add_argument('--epochs', default=50, type=int, help='Max epochs')    # 50
parser.add_argument('--test_freq', default=1, type=int, help='print frequency')
parser.add_argument('--batch', default=2, type=int, help='Batch size')    # 2
parser.add_argument('--test_batch', default=1, type=int, help='Test batch size')   # 1
parser.add_argument('--lr', default=5e-7, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=5e-5, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=45, type=int, help='step down size (cyclic) for learning rate scheduler')#37
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

# GPU
parser.add_argument('--device', default="0,1", type=str, help='GPU device')


# parser.add_argument('--model_path', default="/data/frg/code/tmm/result/test/epoch49_77.13%.pth", type=str, help='pretrained model path')
# parser.add_argument('--model_path', default="/data/frg/code/tmm/result/nba_flipmore0.5_E2longtime/epoch49_77.13%25.pth", type=str, help='pretrained model path')
parser.add_argument('--model_path', default="/data/frg/code/tmm/result/nba_epoch60/epoch49_77.13%.pth", type=str, help='pretrained model path')

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
    exp_name = 'test'
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

    # validate(train_loader, model)
    acc, mean_acc, confusion = validate(test_loader, model)
    print('Accuracy is %.2f' % acc)
    print('Mean accuracy is %.2f' % mean_acc)
    print(confusion)
    # visualize(confusion, '/data/yz/DF-NBA/result/nba-trainenhance-self-DD-2h2h1d-1e-3-ARG71.5/NBA71.5')
    visualize(confusion, '/data/frg/code/tmm/result/test/NBA77.13')


@torch.no_grad()
def validate(test_loader, model):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    accuracies = AverageMeter()
    true = []
    pred = []

    # switch to eval mode
    model.eval()

    for i, (ls, images, activities, label_in, label0, label1, label2, label3) in tqdm(enumerate(test_loader)):
        images = images.cuda()
        activities = activities.cuda()
        label_in = label_in.cuda()
        label0 = label0.cuda()
        label1 = label1.cuda()
        label2 = label2.cuda()

        num_batch = images.shape[0]
        num_frame = images.shape[1]
        activities_in = activities[:, 0].reshape((num_batch,))
        label0 = label0[:, 0].reshape((num_batch,))
        label1 = label1[:, 0].reshape((num_batch,))
        label2 = label2[:, 0].reshape((num_batch,))

        # compute output
        scr, score = model(images)#, output

        criterion1 = torch.nn.MultiLabelSoftMarginLoss()
        loss1 = criterion1(scr, label_in)

        # def resize(img):
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     cur_ratio = img.shape[1] / float(img.shape[0])
        #     target_ratio = 320 / float(180)
        #     mask_h = 180
        #     mask_w = 320
        #     img = np.array(img)
        #     if cur_ratio > target_ratio:
        #         cur_h = 180
        #         cur_w = 320
        #     else:
        #         cur_h = 180
        #         cur_w = int(180 * cur_ratio)
        #     img = cv2.resize(img, (cur_w, cur_h))
        #     start_y = (mask_h - img.shape[0]) // 2
        #     start_x = (mask_w - img.shape[1]) // 2
        #     mask = np.zeros([mask_h, mask_w, 3]).astype(np.uint8)
        #     mask[start_y: start_y + img.shape[0], start_x: start_x + img.shape[1], :] = img
        #     return mask
        #
        # output = output.mean(dim=0)  # .transpose(1, 2)
        #
        # output = output.cpu().numpy()[:, :, :]  # .squeeze(0)
        #
        # for j in range(18):
        #     tang_max_stage1 = output[j, :, :].max()
        #     tang_min_stage1 = output[j, :, :].min()
        #     tang_array_stage1 = (output[j, :, :] - tang_min_stage1) / (
        #             tang_max_stage1 - tang_min_stage1)
        #
        #     # output = tang_array_stage1
        #     [sid, src_fid, fid] = ls[j]
        #     sid = int(sid.numpy())
        #     src_fid = int(src_fid.numpy())
        #     fid = fid.numpy()
        #     fid = '{0:06d}'.format(int(fid))
        #     img = skimage.io.imread("/data/yz/tm-detectorfree/dataset/NBA/NBA_dataset/videos/" + '%d/%d/%s.jpg' % (sid, src_fid, fid))
        #     img_new = resize(img)
        #     heatmap = np.uint8(255 * tang_array_stage1)  # 将热力图转换为RGB格式
        #     heatmap = cv2.resize(heatmap,(320, 180))
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        #     superimposed_img = heatmap * 0.4 + img_new  # 这里的0.4是热力图强度因子
        #     save_img_path_2 = '/data/yz/DF-NBA/result/result/' + '%d_%d_%d_%s.jpg' % (sid, src_fid, j, fid)
        #     cv2.imwrite(save_img_path_2, superimposed_img)  # 将图像保存到硬盘






        # images = images.mean(dim=0)[9, :, :, :]
        # image = PIL.Image.fromarray(torch.clamp(images * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
        # image = torchvision.transforms.functional.to_pil_image(images)
        # #
        # # #visual
        # output = output.mean(dim=0)#.transpose(1, 2)
        #
        # output = output.cpu().numpy()[:, :, :]#.squeeze(0)
        # for j in range(18):
        #     tang_max_stage1 = output[j, :, :].max()
        #     tang_min_stage1 = output[j, :, :].min()
        #     tang_array_stage1 = (output[j, :, :] - tang_min_stage1) / (
        #             tang_max_stage1 - tang_min_stage1)
        #
        #     # output = tang_array_stage1
        #     output1 = Image.fromarray( 150*tang_array_stage1)
        #     if output1.mode == "F":
        #         output1 = output1.convert('RGB')
        #
        #     # output.save('/data/yz/ARG_6/vision/eval_net_ijcai/results/' + str(sid) + '_' + str(src_fid) +  '.jpg')
        #     # print(str(sid) +str(src_fid))
        #     [sid, src_fid, fid] = ls[j]
        #     sid = int(sid.numpy())
        #     src_fid = int(src_fid.numpy())
        #     fid = fid.numpy()
        #     fid = '{0:06d}'.format(int(fid))
        #     img = Image.open("/data/yz/tm-detectorfree/dataset/NBA/NBA_dataset/videos/" + '%d/%d/%s.jpg' % (sid, src_fid, fid))
        #
        #     # img = transforms.functional.resize(img, (args.image_height, args.image_width))
        #     img = transforms.functional.resize(img, (90, 160))
        #
        #     # dataset_path = "/root/autodl-tmp/Datasets/Volleyball_dataset/videos/"
        #     # img_path = dataset_path + '/' + str(sid[0]) + '/' + str(src_fid[0]) + '/' + str(fid[0]) + '.jpg'
        #     # img = Image.open(img_path)
        #
        #     output1 = output1.resize((160, 90))
        #     plt.imshow(img)
        #     # plt.imshow(output1, cmap=plt.get_cmap('jet'), alpha=0.5)
        #     plt.imshow(output1, cmap=plt.get_cmap('pink'), alpha=0.5)
        #
        #     plt.axis("off")
        #
        #     save_img_path_2 = '/data/yz/DF-NBA/result/result/'  + '%d_%d_%d_%s.jpg' % (sid, src_fid, j, fid)
        #     plt.savefig(save_img_path_2, bbox_inches='tight', dpi=600, pad_inches=0.0)
        #     plt.close("all")



        true = true + activities_in.tolist()
        # pred = pred + torch.argmax(score, dim=1).tolist()

        # measure accuracy and record loss
        # group_acc, pred1 = accuracy1(scr, label0, label1)
        group_acc, pred0 = accuracy2(scr, label0, label1, label2)
        pred = pred + pred0
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

# def accuracy2(scr, label0, label1, label2):
    # pred1 = []
    # output0 = torch.argmax(scr[:, :2], dim=1)
    # output1 = torch.argmax(scr[:, 2:4], dim=1)
    # output2 = torch.argmax(scr[:, 4:], dim=1)
    # if output0[0] == 0 and output1[0] == 1 and output2[0] ==0:
    #     pred1 = [0]
    # elif output0[0] == 0 and output1[0] == 1 and output2[0] ==2:
    #     pred1 = [1]
    # elif output0[0] == 0 and output1[0] == 1 and output2[0] ==1:
    #     pred1 = [2]
    # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==0:
    #     pred1 = [3]
    # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==2:
    #     pred1 = [4]
    # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==1:
    #     pred1 = [5]
    # elif output0[0] == 1 and output1[0] == 1 and output2[0] ==0:
    #     pred1 = [6]
    # elif output0[0] == 1 and output1[0] == 1 and output2[0] ==2:
    #     pred1 = [7]
    # elif output0[0] == 1 and output1[0] == 1 and output2[0] ==1:
    #     pred1 = [8]
    # else:
    #     print(label0, label1, label2)
    #     print(output0[0],output1[0],output2[0])
    #     pred1 = [8]
    #     # print(label0, label1, label2)
    # # output3 = torch.argmax(scr[:, 6:], dim=1)
    # tt0 = torch.eq(label0.int(), output0.int())
    # tt1 = torch.eq(label1.int(), output1.int())
    # tt2 = torch.eq(label2.int(), output2.int())
    # # tt3 = torch.eq(label3.int(), output3.int())
    # tt = tt0.int() * tt1.int() * tt2.int()
    # correct = torch.sum(tt).float()
    #
    # # correct = torch.sum(torch.eq(target.int(), output.int())).float()
    # # return correct.item() / output.shape[0]
    # return correct.item() / scr.shape[0], pred1


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
            print(label0, label1, label2)
            print(output0[0],output1[0],output2[0])
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
#     b, c = scr.shape
#     output0 = torch.argmax(scr[:, :2], dim=1)
#     output1 = torch.argmax(scr[:, 2:4], dim=1)
#     scr3 = F.softmax(scr[:, 4:])
#     scr_od = F.softmax(scr_od[:,:])
#     output2 = torch.argmax(scr3[:, :]+scr_od[:,:], dim=1)
#     if output1[0] == 0:
#         if output2[0] ==0:
#             pred1 = [3]
#         elif output2[0] ==2:
#             pred1 = [4]
#         elif output2[0] ==1:
#             pred1 = [5]
#     if output1[0] == 1:
#         if output0[0] == 0 and output2[0] ==0:
#             pred1 = [0]
#         elif output0[0] == 0 and output2[0] ==2:
#             pred1 = [1]
#         elif output0[0] == 0 and output2[0] ==1:
#             pred1 = [2]
#     # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==0:
#     #     pred1 = [3]
#     # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==2:
#     #     pred1 = [4]
#     # elif output0[0] == 0 and output1[0] == 0 and output2[0] ==1:
#     #     pred1 = [5]
#         elif output0[0] == 1 and output2[0] ==0:
#             pred1 = [6]
#         elif output0[0] == 1 and output2[0] ==2:
#             pred1 = [7]
#         elif output0[0] == 1 and output2[0] ==1:
#             pred1 = [8]
#     pred1 = torch.tensor(pred1, dtype=torch.long).cuda()
#     pred1 = pred1.reshape((b,))
#
#     tt0 = torch.eq(label0.int(), output0.int())
#     tt1 = torch.eq(label1.int(), output1.int())
#     tt2 = torch.eq(label2.int(), output2.int())
#
#
#     # tt = tt0.int() * tt1.int() * tt2.int()
#     tt = torch.eq(activities_in.int(), pred1.int())
#     correct = torch.sum(tt).float()
#
#     # correct = torch.sum(torch.eq(target.int(), output.int())).float()
#     # return correct.item() / output.shape[0]
#     return correct.item() / scr.shape[0]

def visualize(matrix, name):
    num_class = 9
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    print(cm)
    plt.figure()  # TITAN可去掉AGG
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(num_class)
    plt.xticks(tick_marks, ('2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.'))
    plt.yticks(tick_marks, ('2p-succ.', '2p-fail.-off.', '2p-fail.-def.',
              '2p-layup-succ.', '2p-layup-fail.-off.', '2p-layup-fail.-def.',
              '3p-succ.', '3p-fail.-off.', '3p-fail.-def.'))

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
