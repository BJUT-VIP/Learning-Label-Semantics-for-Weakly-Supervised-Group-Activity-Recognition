# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/transformer.py
# ------------------------------------------------------------------------
import copy
from typing import Optional
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from collections import OrderedDict
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        # tgt = query_embed
        hs, att = self.decoder(tgt, src, memory_key_padding_mask=mask,
                               pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), att


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        att = None

        intermediate = []
        intermediate_att = []

        for layer in self.layers:
            output, att = layer(output, memory, tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_att.append(att)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_att)

        return output.unsqueeze(0), att.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.TrajectoryAttention = TrajectoryAttention(d_model, nhead, qkv_bias=False, attn_drop=0., proj_drop=0.,
                                                  use_original_code=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # query = self.with_pos_embed(tgt, query_pos)
        # key = self.with_pos_embed(memory, pos)
        # value = memory
        # E = 0
        # memory1, att = self.TrajectoryAttention(memory, attn_mask=memory_mask,
        #                                 key_padding_mask=memory_key_padding_mask)
        # memory = memory + self.dropout0(memory1)
        # memory = self.norm0(memory)

        tgt2, att = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, att = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, att

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_token_encoder(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.drop_rate,
        nhead=args.nheads,
        dim_feedforward=args.ffn_dim,
        num_decoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class Attention_cross(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, edge_importance_weighting=True, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.edge_importance_weighting = edge_importance_weighting

        self.to_qkv_1 = nn.Linear(dim, inner_dim * 2, bias = False)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x, y, mask = None):
        bs, c, h, w = x.shape
        src = x.flatten(2).permute(2, 0, 1)
        b, n, _, h = *x.shape, self.heads
        kv = self.to_qkv_1(x).chunk(2, dim = -1)
        q = self.to_qkv_2(y)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        # q = torch.tenser(map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), q))
        query = q
        key = k.transpose(2,3)
        value = v
        dots = torch.matmul(query,key) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn,value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class TransformerWQ(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.drop_att = nn.Dropout(dropout)
        self.ln_att = nn.LayerNorm([dim])
        self.drop_ffn = nn.Dropout(dropout)
        self.ln_fnn = nn.LayerNorm([dim])


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_cross(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))


    def forward(self, x,y, mask = None):
        for attn, ff in self.layers:
            x = self.ln_att(x + self.drop_att(attn(x ,y,mask=mask)))
            x = self.ln_fnn(x + self.drop_ffn(ff(x,y)))
        return x

class CLIPCAD(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length



        self.transformer = Transformer1(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(self.context_length, transformer_width)#
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.fc_final = nn.Linear(5*256, 256)
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        #
        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # text[:, 0] = 0
        # text[:, 3] = 0
        # text1 = torch.div(text, 6000, out=None)
        # text2 = torch.Tensor(text1.cpu()).long()
        x = self.token_embedding(text.cuda())#.type(self.dtype)  # [batch_size, n_ctx, d_model]

        # x = self.token_embedding(text)

        x = x + self.positional_embedding#.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.fc_final(x.reshape(5,-1))
        x = self.ln_final(x)#.type(self.dtype)


        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        # x = x @ self.text_projection

        return x

    def forward(self, text):

        text_features = self.encode_text(text)

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # shape = [global_batch_size, global_batch_size]
        return text_features

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length



        self.transformer = Transformer1(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(self.context_length, transformer_width)#
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.fc_final = nn.Linear(7*256, 256)
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        #
        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # text[:, 0] = 0
        # text[:, 3] = 0
        # text1 = torch.div(text, 6000, out=None)
        # text2 = torch.Tensor(text1.cpu()).long()
        x = self.token_embedding(text.cuda())#.type(self.dtype)  # [batch_size, n_ctx, d_model]

        # x = self.token_embedding(text)

        x = x + self.positional_embedding#.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.fc_final(x.reshape(9,-1))
        x = self.ln_final(x)#.type(self.dtype)


        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        # x = x @ self.text_projection

        return x

    def forward(self, text):

        text_features = self.encode_text(text)

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # shape = [global_batch_size, global_batch_size]
        return text_features

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class Transformer1(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class TrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, attn_mask=True,
                               key_padding_mask=True):
        _, _, C = x.shape
        P = 920
        # _, F, _ = k.shape#18
        F = 5
        h = self.num_heads

        # k = k.permute(1, 0, 2).reshape(-1,F,P,C).reshape(-1, F * P, C)
        x = x.permute(1, 0, 2).reshape(-1,F,P,C).reshape(-1, F * P, C)
        # if E == 0:
        #     q = q.permute(1, 0, 2).reshape(-1,F,8,C).reshape(-1, F * 8, C)
        # else:
        #     q = q.permute(1, 0, 2).reshape(-1,F,8,C).reshape(-1, F * 8, C)
        B, N, C = x.shape

        # project x to q, k, v vaalues
        q_, k_, v_ = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        # k_, v_ = map(
        #     lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, x))
        # q_ = rearrange(q, 'b n (h d) -> (b h) n d', h=h)


        # remove CLS token from q, k, v
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #     lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        # cls_out = qkv_attn(cls_q * self.scale, k, v)
        # cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)

        q_dot_k = q_ @ k_.transpose(-2, -1)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
        space_attn = (self.scale * q_dot_k).softmax(dim=-1)
        attn = self.attn_drop(space_attn)
        v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)#Bh,Tk,T,D

        #  Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)#B,T,k,T,D
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)#B,T,k,D
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F, h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        # if self.use_original_code:
        #     x = rearrange(x, f'b s f (h d) -> b h s f d', f=F, h=h)
        #     x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        # else:
        #     x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')

        # concat back the cls token
        # x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        # if E == 0:
        #     x = x.reshape(B, F, P, C).reshape(B * F, -1, C).permute(1, 0, 2)
        # else:
        #     x = x.reshape(B, F, P, C).reshape(B * F, -1, C).permute(1, 0, 2)
        # x = x.reshape(-1, F, 6, C).permute(2, 0, 1, 3).reshape(6, -1, C)
        x = x.reshape(B, F, P, C).reshape(B * F, -1, C).permute(1, 0, 2)

        return x, attn