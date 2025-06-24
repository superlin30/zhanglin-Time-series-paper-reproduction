__all__ = ['PDF_backbone2']

import math
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from einops import rearrange
# from collections import OrderedDict
from layers.PDF_layers import *
from layers.RevIN import RevIN
from layers.Conv_Blocks import Inception_Block_V1
from layers.decomp import DECOMP

# Cell
class PDF_backbone2(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int,
                 period, patch_len, stride, kernel_list, serial_conv=False, wo_conv=False, add=False,
                 max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False, 

                 #新增
                 num_kernels:int=8,
                 enc_in:int=7,
                 **kwargs):
        super().__init__()

        self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.period_list = period
        #print(f"PDF_backbone2 period_list {self.period_list}")
        self.period_len = [math.ceil(context_window / i) for i in self.period_list]

        #print(f"PDF_period_len {self.period_len}")

        self.kernel_list = [(n, patch_len[i]) for i, n in enumerate(self.period_len)]
        #print(f"PDF_kernel_list {self.kernel_list}")    

        self.stride_list = [(n , m // 2 if stride is None else stride[i]) for i, (n, m) in enumerate(self.kernel_list)]
        #print(f"PDF_stride_list {self.stride_list}")

        self.dim_list = [k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]

        self.pad_layer = nn.ModuleList([nn.ModuleList([
            nn.ConstantPad1d((0, p-context_window%p), 0)if context_window % p != 0 else nn.Identity(),
            nn.ConstantPad1d((0, k[1] - p % k[1]), 0) if p % k[1] != 0 else nn.Identity()
        ]) for p, (k, s) in zip(self.period_list, zip(self.kernel_list, self.stride_list))
        ])
        #print(f"PDF_pad_layer {self.pad_layer}")

        self.embedding = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s),
            nn.Flatten(start_dim=2)
        ) for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ])
        #print(f"PDF_embedding_layer {self.embedding}")

        self.backbone = nn.ModuleList([nn.Sequential(
            TSTiEncoder(c_in, patch_num=token, patch_len=self.dim_list[i], max_seq_len=max_seq_len,
                        n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                        norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                        key_padding_mask=key_padding_mask, padding_var=padding_var,
                        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                        store_attn=store_attn,
                        pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs),
            nn.Flatten(start_dim=-2),
            nn.Linear(self.tokens_list[i] * d_model, context_window)
            if self.tokens_list[i] * d_model != context_window else nn.Identity()
        ) for i, token in enumerate(self.tokens_list)])
        
        self.wo_conv = wo_conv
        self.serial_conv = serial_conv

        self.head = Head(context_window, len(period), target_window, head_dropout=head_dropout, Concat=not add)

        

        
        # 原固定感受野
        if not self.wo_conv:
            self.conv = nn.ModuleList([nn.Sequential(*[
                nn.Sequential(nn.Conv1d(n, n, kernel_size=i, groups=n, padding=i//2), nn.SELU())
                for i in kernel_list],
                nn.Dropout(fc_dropout),
                nn.Flatten(start_dim=-2),
            ) for n in self.period_len])
        
        '''
        # 自适应感受野
        if not self.wo_conv:
            self.conv = nn.ModuleList([
                ERFAdaptiveConvBlock(n, period_len=period_len_i)
                for n, period_len_i in zip(self.period_len, self.period_len)
            ])
        '''
        
        # 新增x patch
        """
        self.use_decomp = True  # 设为可配置
        self.decomp = DECOMP(ma_type='dema', alpha=0.3, beta=0.3)  # x patch参数
        """



 




    def forward(self, z):  # z: [bs x nvars x seq_len]

        #print(f"进入PDF_backbone2后 {z.shape}")
        # norm
        z = z.permute(0, 2, 1)
        #print(f"进入PDF_backbone2后移动 {z.shape}")
        z = self.revin_layer(z, 'norm')

        #print(f"revin_layer归一化之后 {z.shape}")

        """ 
        if self.use_decomp:
            seasonal, trend = self.decomp(z)
            seasonal = seasonal.permute(0, 2, 1)  # [B, C, T]
            trend = trend.permute(0, 2, 1)
        else:
            seasonal = z.permute(0, 2, 1)
            trend = torch.zeros_like(seasonal)

        """ 


        z = z.permute(0, 2, 1)
        #print(f"进入PDF_backbone2后，revin_layer后，移动回来 {z.shape}")


        res = []


        
        # 原PDF 长短
        if self.wo_conv:
            for i, period in enumerate(self.period_list):
                glo = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                glo = self.pad_layer[i][1](glo)
                glo = self.embedding[i](glo.unsqueeze(-3))
                glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
                glo = self.backbone[i](glo)
                res.append(glo)
        elif self.serial_conv:
            for i, period in enumerate(self.period_list):
                loc = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                loc = self.conv[i](loc).reshape(z.shape[0], z.shape[1], -1)[..., :z.shape[-1]]
                glo = self.pad_layer[i][0](loc).reshape(z.shape[0] * z.shape[1], -1, period)
                glo = self.pad_layer[i][1](glo)
                glo = self.embedding[i](glo.unsqueeze(-3))
                glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
                glo = self.backbone[i](glo)
                res.append(glo)
        else:
            for i, period in enumerate(self.period_list):
                #print(f"结合卷积和周期性解耦,将数据分为多个周期 {self.period_list}")
               
                glo = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                #print(f"周期性解耦后glo shape {glo.shape}")

                loc = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                #print(f"周期性解耦后loc shape {loc.shape}")

                loc = self.conv[i](loc).reshape(z.shape[0], z.shape[1], -1)[..., :z.shape[-1]]
                #print(f"卷积捕获短期依赖loc shape {loc.shape}")

                glo = self.pad_layer[i][1](glo)
                #print(f"第二次填充处理 glo shape {glo.shape}")

                #print(f"glo.unsqueeze(-3) {glo.unsqueeze(-3).shape}")

                glo = self.embedding[i](glo.unsqueeze(-3))
                #print(f"glo.unsqueeze(-3) 输入到嵌入层 self.embedding[i]，并调整 glo 的形状 {glo.shape}")

                glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
                #print(f"调整数据的维度，使其符合 Transformer 编码器的输入要求。 {glo.shape}")

                
                glo = self.backbone[i](glo)
                #print(f"将调整后的 glo 输入到 backbone 层进行编码。。 {glo.shape}")

                res.append(glo + loc)
                #print(f"len(res) {len(res)}")
                #print(f"res.append(glo + loc) res[0]shape {res[0].shape}")
        
        """ 
        # 分离trend
        for i, period in enumerate(self.period_list):
            glo = self.pad_layer[i][0](seasonal).reshape(seasonal.shape[0] * seasonal.shape[1], -1, period)
            loc = self.pad_layer[i][0](seasonal).reshape(seasonal.shape[0] * seasonal.shape[1], -1, period)
            loc = self.conv[i](loc).reshape(seasonal.shape[0], seasonal.shape[1], -1)[..., :seasonal.shape[-1]]
            glo = self.pad_layer[i][1](glo)
            glo = self.embedding[i](glo.unsqueeze(-3))
            glo = rearrange(glo, "(b m) d n -> b m d n", b=seasonal.shape[0]).contiguous()
            glo = self.backbone[i](glo)
            res.append(glo + loc)
        """

        # denorm
        z = self.head(res)

        """
        if self.use_decomp:   # 加入use_decomp
            trend_out = trend[..., -z.shape[-1]:]
            z = z + trend_out
        """

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')
        z = z.permute(0, 2, 1)
        return z

class Head(nn.Module):
    def __init__(self, context_window, num_period, target_window, head_dropout=0,
                 Concat=True):
        super().__init__()
        self.Concat = Concat
        self.linear = nn.Linear(context_window * (num_period if Concat else 1), target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.Concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x

class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn, pos=self.W_pos)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                 pos=None
                 ):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn, pos=pos) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False, pos=None):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention, pos=pos)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False, pos=None):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.pos = pos
        self.P_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.P_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        q_p = self.P_Q(self.pos).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_p = self.P_K(self.pos).view(1, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                              q_p=q_p, k_p=k_p)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor((head_dim * 1) ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,
                q_p=None, k_p=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]
        # attn_scores += torch.matmul(q_p, k) * self.scale
        # attn_scores += torch.matmul(q, k_p) * self.scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
        

# 新增，自适应感受野
class ERFAdaptiveConvBlock(nn.Module):
    def __init__(self, in_channels, period_len, kernel_list=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=k, dilation=period_len // k,
                      padding=(period_len // k) * (k // 2), groups=in_channels)
            for k in kernel_list
        ])
        self.norm = nn.BatchNorm1d(in_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B*N, C, L]
        out = sum([block(x) for block in self.blocks]) / len(self.blocks)
        out = self.act(self.norm(out))
        return self.dropout(out + x)  # 残差连接






