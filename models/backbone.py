# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from spatial_correlation_sampler import SpatialCorrelationSampler

from .position_encoding import build_position_encoding


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torchvision.models as models
from .backbone_inv3 import MyInception_v3, MyRes18

class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        #backbone = getattr(torchvision.models, args.backbone)(
         #   replace_stride_with_dilation=[False, False, args.dilation], pretrained=True)

        backbone = models.resnet18(pretrained = True)##############my


        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if self.motion:
            self.layer_channel = [64, 128, 256, 512]

            self.channel_dim = self.layer_channel[self.motion_layer - 1]

            self.corr_input_proj = nn.Sequential(
                nn.Conv2d(self.channel_dim, self.corr_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

            self.neighbor_size = args.neighbor_size
            self.ps = 2 * args.neighbor_size + 1

            self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                                 stride=1, padding=0, dilation_patch=1)

            self.corr_output_proj = nn.Sequential(
                nn.Conv2d(self.ps * self.ps, self.channel_dim, kernel_size=1, bias=False),
                nn.ReLU()
            )

    def get_local_corr(self, x):
        x = self.corr_input_proj(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)

        corr = self.corr_output_proj(corr)

        return corr

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if self.motion:
            if self.motion_layer == 1:
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 2:
                x = self.layer2(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer3(x)
                x = self.layer4(x)
            elif self.motion_layer == 3:
                x = self.layer2(x)
                x = self.layer3(x)
                corr = self.get_local_corr(x)
                x = x + corr
                x = self.layer4(x)
            elif self.motion_layer == 4:
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                corr = self.get_local_corr(x)
                x = x + corr
            else:
                assert False
        else:
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x


class MultiCorrBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args):
        super(MultiCorrBackbone, self).__init__()

        backbone = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation],
            pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.motion = args.motion
        self.motion_layer = args.motion_layer
        self.corr_dim = args.corr_dim

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer_channel = [64, 128, 256, 512]

        self.channel_dim = self.layer_channel[self.motion_layer - 1]

        self.corr_input_proj1 = nn.Sequential(
            nn.Conv2d(self.layer_channel[2], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_input_proj2 = nn.Sequential(
            nn.Conv2d(self.layer_channel[3], self.corr_dim, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.neighbor_size = args.neighbor_size
        self.ps = 2 * args.neighbor_size + 1

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=self.ps,
                                                             stride=1, padding=0, dilation_patch=1)

        self.corr_output_proj1 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[2], kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.corr_output_proj2 = nn.Sequential(
            nn.Conv2d(self.ps * self.ps, self.layer_channel[3], kernel_size=1, bias=False),
            nn.ReLU()
        )

    def get_local_corr(self, x, idx):
        if idx == 0:
            x = self.corr_input_proj1(x)
        else:
            x = self.corr_input_proj2(x)
        x = F.normalize(x, dim=1)
        x = x.reshape((-1, self.num_frames) + x.size()[1:])
        b, t, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()                                       # [B, C, T, H, W]

        # new implementation
        x_pre = x[:, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x_post = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        corr = self.correlation_sampler(x_pre, x_post)                                  # [B x T, P, P, H, W]
        corr = corr.view(-1, self.ps * self.ps, h, w)                                   # [B x T, P * P, H, W]
        corr = F.relu(corr)

        if idx == 0:
            corr = self.corr_output_proj1(corr)
        else:
            corr = self.corr_output_proj2(corr)

        return corr

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        corr = self.get_local_corr(x, 0)
        x = x + corr

        x = self.layer4(x)
        corr = self.get_local_corr(x, 1)
        x = x + corr

        return x


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        features = self[0](x)
        pos = self[1](features).to(x.dtype)

        return features, pos

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # self.l = nn.Linear(441*1024,  1024)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = x[:, 1:]
        x = x.mean(dim = 1)
        x = self.to_latent(x)
        return self.mlp_head(x)

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # position_embedding = 0
    if args.multi_corr:
        backbone = MultiCorrBackbone(args)
    else:
        backbone = Backbone(args)
        # backbone = MyInception_v3(transform_input=False, pretrained=True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def VIT_backbone():
    v = ViT(
        image_size = 336,
        patch_size = 16,
        num_classes = 8,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    model = v

    return model


