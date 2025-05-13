import sys
from torchstat import stat
from torchsummary import summary
from thop import profile, clever_format
from models.models_ev import *
# import models.models_nba as models

import argparse

parser = argparse.ArgumentParser(description='Detector-Free Weakly Supervised Group Activity Recognition')

# Dataset specification
parser.add_argument('--dataset', default='nba', type=str, help='volleyball or nba')
parser.add_argument('--data_path', default='/data/yz/tm-detectorfree/dataset/NBA/', type=str, help='data path')
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=18
                    , type=int, help='number of frames for each clip')
parser.add_argument('--num_total_frame', default=72, type=int, help='number of total frames for each clip')
parser.add_argument('--num_activities', default=9, type=int, help='number of activity classes in volleyball dataset')

# Model parameters
parser.add_argument('--base_model', action='store_true', help='average pooling base model')
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')
parser.add_argument('--dilation', action='store_true', help='use dilation or not')
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

# Motion parameters
parser.add_argument('--motion', default=False, help='use motion feature computation')
parser.add_argument('--multi_corr', default=False, help='motion correlation block at 4th and 5th')
parser.add_argument('--motion_layer', default=4, type=int, help='backbone layer for calculating correlation')
parser.add_argument('--corr_dim', default=64, type=int, help='projection for correlation computation dimension')
parser.add_argument('--neighbor_size', default=5, type=int, help='correlation neighborhood size')

# Transformer parameters
parser.add_argument('--nheads', default=2, type=int, help='number of heads')
parser.add_argument('--enc_layers', default=1, type=int, help='number of encoder layers')
parser.add_argument('--pre_norm', action='store_true', help='pre normalization')
parser.add_argument('--ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_tokens', default=9, type=int, help='number of queries')

# Aggregation parameters
parser.add_argument('--nheads_agg', default=2, type=int, help='number of heads for partial context aggregation')

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=30, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=1, type=int, help='print frequency')
parser.add_argument('--batch', default=2, type=int, help='Batch size')
parser.add_argument('--test_batch', default=1, type=int, help='Test batch size')
parser.add_argument('--lr', default=1e-6, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=5e-5, type=float, help='Max learning rate')
parser.add_argument('--lr_step', default=5, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')

# GPU
parser.add_argument('--device', default="3", type=str, help='GPU device')
# Load model
parser.add_argument('--load_model', action='store_true', help='load model')
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

args = parser.parse_args()
model = DFGAR(args)
# stat(model, (18*3, 720, 1280))
#
input = torch.randn(1, 3*18, 720, 1280)  # batchsize=1, 输入向量长度为10
macs, params = profile(model, inputs=(input, ))
print(' FLOPs: ', macs*2)   # 一般来讲，FLOPs是macs的两倍
print('params: ', params)

print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")
print("%s | %.2f | %.2f" % ("DF", params / (1000 ** 2), (macs * 2) / (1000 ** 3)))