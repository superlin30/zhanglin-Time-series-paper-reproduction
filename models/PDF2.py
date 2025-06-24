__all__ = ['PDF2']

# Cell
from typing import Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from layers.PDF_backbone import PDF_backbone
from layers.PDF_backbone2 import PDF_backbone2
from layers.Conv_Blocks import Inception_Block_V1


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        add = configs.add
        wo_conv = configs.wo_conv
        serial_conv = configs.serial_conv

        kernel_list = configs.kernel_list
        patch_len = configs.patch_len
        period = configs.period
        stride = configs.stride

        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        # 新增
        num_kernels = configs.num_kernels
        batch = configs.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # model
        self.model = PDF_backbone2(c_in=c_in, context_window=context_window, target_window=target_window,
                                           wo_conv=wo_conv, serial_conv=serial_conv, add=add,
                                           patch_len=patch_len, kernel_list=kernel_list, period=period, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                           head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)
        

        # 新增
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.enc_in, configs.d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.enc_in,
                               num_kernels=num_kernels))
        #self.weights = nn.Parameter(torch.randn(1, 3).to(self.device))  # 用 (1, 2) 初始化，不绑定 batch_siz,# 第四版本改成1试一试,试试3
        self.weights = nn.Parameter(torch.randn(batch, 2))  # Random initialization
        
        self.heatmap_to_pred = nn.Conv1d(in_channels=context_window, out_channels=target_window, kernel_size=1)

        # 新增
        self.gate_fusion = nn.Sequential(nn.Linear(2, 2),   # 跨一阶/二阶通道做映射
                                         nn.Sigmoid()
                                        )


    def forward(self, x):  # x: [Batch, Input length, Channel]
        #print(f"移动前 {x.shape}")

        B, T, N = x.size() # [Batch, Input length, Channel]

        #print('-------------------分隔,进入heatmap')

        # heatmap
        heatmap  = compute_derivative_heatmaps(x).permute(0,1,2,3)                          # [B, T, N, 2]
        #print(f"通过一阶二阶导数compute_derivative_heatmaps(x)",compute_derivative_heatmaps(x).size())
        #print(f"heatmap通过一阶二阶导数compute_derivative_heatmaps(x).permute(0,1,2,3)",heatmap.size())

        
        heatmap_features = self.conv(heatmap).permute(0,2,1,3)                      # torch.Size([B, T, N, 2])
        #print(f"heatmap通过conv卷积之后heatmap_features.permute(0,2,1,3).shape()",heatmap_features.size())
        '''
        
        if self.data == 'm4':
            weights = nn.Parameter(torch.randn(B, 2)).to(self.device)  # Random initialization
            weights = weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        else:   
            weights = self.weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1) 
        '''

        # 1.weights方法
        #固定batch_size
        weights = self.weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1) 

        # 动态调整 weights 的形状，确保它与当前的 batch_size 匹配
        #weights = self.weights.expand(B, -1)  # 将权重扩展为 [B, 2]
        #weights = weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1).to(self.device)  # 扩展为 [B, T, N, 2
  
        #print(f"通过weights.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)",weights.size())   

        Final_heatmap = torch.sum(heatmap_features * weights, -1)           #  torch.Size([B, T, N])

        # 2.gate方法
        # [B, T, N, 2] → [B, T, N, 2]
        #gate = self.gate_fusion(heatmap_features)  

        # gated fusion: element-wise × and sum over derivative dimension
        #Final_heatmap = torch.sum(gate * heatmap_features, dim=-1)  # [B, T, N]



        Final_heatmap = Final_heatmap.permute(0,2,1)   
        #print(f"第一个Final_heatmap",Final_heatmap.size())
        #  torch.Size([B,N, T])
        # Transform heatmap to match Pred_len
        Final_heatmap = Final_heatmap.permute(0, 2, 1)  # [B, T, N]
        Final_heatmap = self.heatmap_to_pred(Final_heatmap)  # [B, Pred_len, N]
        #print(f"经过heatmap_to_pred的Final_heatmap",Final_heatmap.size())
        Final_heatmap = Final_heatmap.permute(0, 2, 1)  # [B, N, Pred_len]
        
        #print('-------------------分隔,Final_heatmap')
        


        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        #将数据的特征维度（Channel）移到第二个位置
        #print(f"移动后 {x.shape}")


        #print("-------------------------分隔，进入经过PDF_backbone后")
        x = self.model(x)
        #print("-------------------------分隔，结束进入")

        #print(f"经过PDF_backbone后 {x.shape}")

        combined  = x  + Final_heatmap

        combined = combined.permute(0, 2, 1)  # combined: [Batch, Input length, Channel]
        #print(f"调回来最终combined {combined.shape}")
        return combined
    

def compute_derivative_heatmaps(x):
    
    # Calculate first derivative manually
    first_derivative = x[:, 1:] - x[:, :-1]
    # Pad the first_derivative to maintain original length
    first_derivative = torch.cat([torch.zeros_like(first_derivative[:, :1, :]), first_derivative], dim=1)
    
    # Calculate second derivative manually
    second_derivative = first_derivative[:, 1:] - first_derivative[:, :-1]
    # Pad the second_derivative to maintain original length
    second_derivative = torch.cat([torch.zeros_like(second_derivative[:, :1, :]), second_derivative], dim=1)

    # Stack the derivatives along a new dimension
    heatmap = torch.stack([second_derivative], dim=-1)  # 只保留第二个维度
    #heatmap = torch.stack([first_derivative, second_derivative], dim=-1)
    heatmap = heatmap.permute(0,2,1,3)  # Adjust shape to [Batch, Channels, Time, Derivative]
    return heatmap


    

