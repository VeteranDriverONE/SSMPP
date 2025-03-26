from audioop import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import unfoldNd

from torch.distributions.normal import Normal
from torchvision import transforms
from einops import rearrange
from einops.layers.torch import Rearrange

from MS_models.utils.metrics import ContrastiveLoss
from MS_models.ConvBlock2 import DoubleConv, normalize

class DoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, non_line_op=nn.LeakyReLU,
        conv_args={'kernel_size':3,'stride':1,'padding':1,'dilation': 1, 'bias': True}, conv_args2=None,
        norm_op_args={'eps': 1e-5, 'affine': True, 'momentum': 0.1},
        non_line_op_args={'negative_slope': 1e-2, 'inplace': True},
        first_op=None, first_op_args=None,
        last_op=None, last_op_args=None, DIM=2):

        super(DoubleConv, self).__init__()

        self.first_op = first_op
        self.last_op = last_op

        if conv_args2 is None:
            conv_args2 = conv_args

        conv_op = getattr(nn, f'Conv{DIM}d')
        norm_op = getattr(nn, f'BatchNorm{DIM}d')

        self.conv1 = conv_op(in_channel, mid_channel, **conv_args)
        self.norm1 = norm_op(mid_channel, **norm_op_args)
        self.non_line1 = non_line_op(**non_line_op_args)

        self.conv2 = conv_op(mid_channel, out_channel, **conv_args2)
        self.norm2 = norm_op(out_channel, **norm_op_args)
        self.non_line2 = non_line_op(**non_line_op_args)
        
        if first_op is not None:
            self.first_op = first_op(**first_op_args)

        if last_op is not None:
            self.last_op = last_op(**last_op_args)

    def forward(self,x):
        if self.first_op is not None:
            x = self.first_op(x)

        x1 = self.non_line1(self.norm1(self.conv1(x)))
        x2 = self.non_line2(self.norm2(self.conv2(x1)))

        if self.last_op is not None:
            x2 = self.last_op(x2)

        return x2
    
class OutDoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, non_line_op=nn.LeakyReLU,
        conv_args={'kernel_size':3,'stride':1,'padding':1,'dilation': 1, 'bias': True}, conv_args2=None,
        norm_op_args={'eps': 1e-5, 'affine': True, 'momentum': 0.1},
        non_line_op_args={'negative_slope': 1e-2, 'inplace': True},
        first_op=None, first_op_args=None,
        last_op=None, last_op_args=None, DIM=2):

        super(OutDoubleConv, self).__init__()

        self.first_op = first_op
        self.last_op = last_op

        if conv_args2 is None:
            conv_args2 = conv_args

        conv_op = getattr(nn, f'Conv{DIM}d')
        norm_op = getattr(nn, f'BatchNorm{DIM}d')

        self.conv1 = conv_op(in_channel, mid_channel, **conv_args)
        self.norm1 = norm_op(mid_channel, **norm_op_args)
        self.non_line1 = non_line_op(**non_line_op_args)

        self.conv2 = conv_op(mid_channel, out_channel, **conv_args2)
        self.norm2 = norm_op(out_channel, **norm_op_args)
        self.non_line2 = nn.Sigmoid()
        
        if first_op is not None:
            self.first_op = first_op(**first_op_args)

        if last_op is not None:
            self.last_op = last_op(**last_op_args)

    def forward(self,x):
        if self.first_op is not None:
            x = self.first_op(x)

        x1 = self.non_line1(self.norm1(self.conv1(x)))
        x2 = self.non_line2(self.norm2(self.conv2(x1)))

        if self.last_op is not None:
            x2 = self.last_op(x2)

        return x2

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1) # 按最后一维划分变成三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ConvEncoder(nn.Module):
    def __init__(self, inshape, in_ch=1, stage_num=4,  conv_pool=False, pool_args=None,  freeze=False):
        super(ConvEncoder, self).__init__()
        DIM = len(inshape)
        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        
        if conv_pool:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args, DIM=DIM),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args, DIM=DIM),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args, DIM=DIM),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args,DIM=DIM),
                # DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, DIM=DIM),
                # DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
        else:
            pool_op = getattr(nn,f'MaxPool{DIM}d')
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                # DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, DIM=DIM),
                # DoubleConv(8*base_ch, 16*base_ch, 16*base_ch, DIM=DIM),
            ]

        self.encode_blocks = nn.ModuleList(encode_blocks[:stage_num])
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1), margin=2)

        prototypes = []
        inputs = torch.rand(inshape).unsqueeze(0).unsqueeze(0)
        for d in range(self.stage_num):
            inputs = self.encode_blocks[d](inputs)
            prototypes.append(nn.Parameter(torch.rand(inputs.shape[1])))
        
        self.prototypes = nn.ParameterList(prototypes)

    def forward1(self, x, x_mask=None):
        # X: B,C,H,W
        # x_mask, B,1,H,W
        skips =[]
        if x_mask is not None:
            contra_loss = torch.tensor(0)
            for d in range(self.stage_num):
                x = self.encode_blocks[d](x)
                skips.append(x)
                x_mask = nn.functional.interpolate(x_mask, scale_factor=0.5, mode='bilinear').round()
                
                if x_mask.sum() != 0:
                    contra_loss = contra_loss + self.cal_contra(x, x_mask)

            return  skips, contra_loss
        
        else:
            for d in range(self.stage_num):
                x = self.encode_blocks[d](x)
                skips.append(x)
            return skips

    def forward2(self, x1, x_mask1, x2, x_mask2):
        # encoder agai
        contra_loss = torch.tensor(0)

        for d in range(self.stage_num):
            x1 = self.encode_blocks[d](x1)
            x2 = self.encode_blocks[d](x2)
            
            x_mask1 = nn.functional.interpolate(x_mask1, scale_factor=0.5, mode='bilinear').round()
            x_mask2 = nn.functional.interpolate(x_mask2, scale_factor=0.5, mode='bilinear').round()

            if x_mask1.sum()==0 or x_mask2.sum() == 0:
                break
            
            contra_loss = contra_loss + self.cal_contra2(x1, x_mask1, x2, x_mask2)
            
        return contra_loss
    
    def cal_contra(self, x, mask):
        # x1: B,C,H,W; mask1:B,1,H,W
        x = x.permute(0,2,3,1).flatten(end_dim=-2)
        mask = mask.permute(0,2,3,1).flatten(end_dim=-2).squeeze()
        
        index = torch.nonzero(mask)

        idx = torch.randperm(index.shape[0])
        pos_index = index[idx,:].view(index.size())

        re_index = torch.nonzero(1-mask)
        idx = torch.randperm(re_index.shape[0])
        neg_index = re_index[idx,:].view(re_index.size())
        neg_index = neg_index[:len(pos_index)]
        
        new_x = torch.index_select(x, 0, index.squeeze())
        pos_x = torch.index_select(x, 0, pos_index.squeeze())
        neg_x = torch.index_select(x, 0, neg_index.squeeze())

        triplet_loss = self.triplet_loss(new_x, pos_x, neg_x)

        return triplet_loss
    
    def cal_contra2(self, x1, mask1, x2, mask2):
        # x1: B,C,H,W; mask1:B,1,H,W
        x1 = x1.permute(0,2,3,1).flatten(end_dim=-2)
        mask1 = mask1.permute(0,2,3,1).flatten(end_dim=-2).squeeze()

        x2 = x2.permute(0,2,3,1).flatten(end_dim=-2)
        mask2 = mask2.permute(0,2,3,1).flatten(end_dim=-2).squeeze()
        
        index1 = torch.nonzero(mask1)
        index2 = torch.nonzero(mask2)

        max_emb = min(len(index1), len(index2))

        index1 = index1[:max_emb]
        index2 = index2[:max_emb]

        re_index = torch.nonzero(1-mask1)
        idx = torch.randperm(re_index.shape[0])
        neg_index = re_index[idx,:].view(re_index.size())
        neg_index = neg_index[:max_emb]
        
        x2 = mask2.unsqueeze(-1) * x2

        new_x = torch.index_select(x1, 0, index1.squeeze())
        pos_x = torch.index_select(x2, 0, index2.squeeze())
        neg_x = torch.index_select(x1, 0, neg_index.squeeze()) 

        # return torch.nn.functional.triplet_margin_loss(new_x, pos_x, neg_x, margin=5, reduction='mean')
        return self.triplet_loss(new_x, pos_x, neg_x)

    def un_contra_loss(self, skips, x_mask1, un_skips, x_mask2):
        
        contra_loss = torch.tensor(0)
        for i in range(len(skips)):
            x_mask1 = nn.functional.interpolate(x_mask1, scale_factor=0.5, mode='bilinear').round()
            x_mask2 = nn.functional.interpolate(x_mask2, scale_factor=0.5, mode='bilinear').round()

            if x_mask1.sum()==0 or x_mask2.sum() == 0:
                break
            
            contra_loss = contra_loss + self.cal_contra2(skips[i], x_mask1, un_skips[i], x_mask2)
            
        return contra_loss


class Decoder(nn.Module):
    # 采用风格
    def __init__(self, inshape, out_ch=1, stage_num=4, conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Decoder, self).__init__()
        DIM = len(inshape)
        interpolation_mode = "bilinear" if DIM==2 else "trilinear"

        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool
        self.up_sample = up_sample

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'scale_factor':2, 'mode':'nearest', 'align_corners':None}

        decode_blocks = [
            # DoubleConv(16*base_ch, 16*base_ch, 8*base_ch),
            DoubleConv(1*8*base_ch, 8*base_ch, 4*base_ch,  DIM=DIM),
            DoubleConv(2*4*base_ch, 4*base_ch, 2*base_ch, DIM=DIM),
            DoubleConv(2*2*base_ch, 2*base_ch, base_ch, DIM=DIM),
            OutDoubleConv(2*base_ch, base_ch, out_ch, DIM=DIM)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            ConTranspose_op = getattr(nn, f'ConvTranspose{DIM}d')
            self.up_sample = [
                ConTranspose_op(1*8*base_ch, 1*8*base_ch, 2, 2),
                ConTranspose_op(2*4*base_ch, 2*4*base_ch, 2, 2),
                ConTranspose_op(2*2*base_ch, 2*2*base_ch, 2, 2),
                ConTranspose_op(2*base_ch, 2*base_ch, 2, 2),
            ]

        self.decode_blocks = nn.ModuleList(decode_blocks)
        self.up_sample = nn.ModuleList(self.up_sample)

        conv_fn = getattr(nn, f'Conv{DIM}d')
        out_up = [
            # nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            nn.Upsample(scale_factor=8, mode=interpolation_mode, align_corners=True),
            nn.Upsample(scale_factor=4, mode=interpolation_mode, align_corners=True),
            nn.Upsample(scale_factor=2, mode=interpolation_mode, align_corners=True),
        ]
        out_conv = [
            # conv_fn(8*base_ch, 1, kernel_size=1),
            conv_fn(4*base_ch, out_ch, kernel_size=1, bias=False),
            conv_fn(2*base_ch, out_ch, kernel_size=1, bias=False),
            conv_fn(base_ch, out_ch, kernel_size=1, bias=False),
            # conv_fn(base_ch, out_ch, kernel_size=1),
        ]

        self.out_up = nn.ModuleList(out_up)
        self.out_conv = nn.ModuleList(out_conv)

    def forward(self, skips):
        outputs = []
        d_x = self.up_sample[0](skips[-1])
        d_x = self.decode_blocks[0](d_x)
        for u in range(1, len(self.decode_blocks)):

            tmp_x = self.out_conv[u-1](self.out_up[u-1](d_x))
            outputs.append(torch.sigmoid(tmp_x)) # 中间输出

            d_x = torch.concat([d_x, skips[-(u+1)]], dim=1)
            d_x = self.decode_blocks[u](self.up_sample[u](d_x))

        outputs.append(d_x)
        return outputs


class Atts(nn.Module):
    def __init__(self, T_num, layer_shapes):
        super(Atts, self).__init__()
        self.T_num = T_num
        atts = []

        for i in range(T_num):
            # atts.append(MultiLayerAttention(layer_shapes))
            atts.append(SparseMultiLayerAttention(layer_shapes))
            # atts.append(SparseMultiLayerAttention2(layer_shapes))

        self.atts = nn.ModuleList(atts)

    def forward(self, x_skips):
        # F1, F2, F3, F4
        # B,C,H,W; B,2C,H/2,E/2; B,4C,H/4,W/4; B,8C,H/8,W/8；B,8C,H/16,W/16；
        for i in range(self.T_num):
            x_skips = self.atts[i](x_skips)

        return x_skips


class MultiLayerAttention(nn.Module):
    def __init__(self, layer_shapes:list):
        super().__init__()
        norms = []
        att_blocks = []
        self.layer_shapes = layer_shapes
        for i in range(len(layer_shapes)):
            # norms.append(nn.LayerNorm(layer_shapes[i][1]))
            att_blocks.append(Attention2(Q_dim=layer_shapes[i][1], KV_dim=[layer_shapes[j][1] for j in range(len(layer_shapes)) if j!=i], heads=4, dim_head=layer_shapes[i][1]))
            # att_blocks.append(Attention3(dim_list=[layer_shapes[j][1] for j in range(len(layer_shapes))], kv_index=i, heads=4, dim_head=layer_shapes[i][1]))

        self.att_blocks = nn.ModuleList(att_blocks)
        # self.norms = nn.ModuleList(norms)
    
    # def forward(self, x_skips):
        
    #     tmp_skips = []
    #     new_skips = []
    #     B = x_skips[0].shape[0]
    #     # 转换通道，并正则化
    #     for i in range(len(x_skips)):
    #         tmp = x_skips[i].permute(0,2,3,1).reshape(B, -1, self.layer_shapes[i][1]) # B,N,D
    #         tmp_skips.append(self.norms[i](tmp))
        
    #     for index in range(len(self.att_blocks)):
    #         new_skips.append(self.att_blocks[index](tmp_skips, index))

    #     for i in range(len(new_skips)):
    #         tmp = new_skips[i]
    #         tmp = tmp.permute(0,2,1).reshape([B]+self.layer_shapes[i][1:]) # reshape B,C,H,W
    #         new_skips[i] = tmp

    #     return new_skips

    def forward(self, x_skips):
        
        tmp_skips = []
        new_skips = []
        B = x_skips[0].shape[0]

        # 转换通道，并正则化
        for i in range(len(self.att_blocks)):
            tmp_skips.append( x_skips[i].permute(0,2,3,1).reshape(B, -1, self.layer_shapes[i][1]) ) # B,N,D
        
        for index in range(len(self.att_blocks)):
            if index <= 5:
                new_skips.append(self.att_blocks[index](tmp_skips, index))
            else:
                new_skips.append(tmp_skips[index])

        for i in range(len(new_skips)):
            new_skips[i] = new_skips[i].permute(0,2,1).reshape([B]+self.layer_shapes[i][1:]) # reshape B,C,H,W

        return new_skips

# 改
class SparseMultiLayerAttention(nn.Module):
    def __init__(self, layer_shapes:list):
        super().__init__()
        norms = []
        att_blocks = []
        unfold = []
        top_k = []
        ks=[(7,7),(5,5),(3,3),(1,1)]
        ss=[(7,7),(5,5),(3,3),(1,1)]
        top_k = [15,10,4,1]
        self.layer_shapes = layer_shapes
        for i in range(len(layer_shapes)):
            # norms.append(nn.LayerNorm(layer_shapes[i][1]))
            # att_blocks.append(Attention2(Q_dim=layer_shapes[i][1], KV_dim=[layer_shapes[j][1] for j in range(len(layer_shapes)) if j!=i], heads=4, dim_head=layer_shapes[i][1]))
            att_blocks.append(Attention4(Q_dim=layer_shapes[i][1], KV_dim=[layer_shapes[j][1] for j in range(len(layer_shapes)) if j!=i], heads=4, dim_head=layer_shapes[i][1]))
            unfold.append(unfoldNd.UnfoldNd(kernel_size=ks[i], stride=ss[i]))
            top_k[i] = np.ceil(np.prod(ks[i])/top_k[i]).astype(np.uint8)
            # att_blocks.append(Attention3(dim_list=[layer_shapes[j][1] for j in range(len(layer_shapes))], kv_index=i, heads=4, dim_head=layer_shapes[i][1]))

        self.att_blocks = nn.ModuleList(att_blocks)
        self.unfolds = nn.ModuleList(unfold)
        self.top_k = top_k
        # self.norms = nn.ModuleList(norms)

    def forward(self, x_skips):
        tmp_Qs =[]
        tmp_skips = []
        new_skips = []
        B = x_skips[0].shape[0]
        # 转换通道，并正则化
        for i in range(len(self.att_blocks)):
            # tmp_skips.append( x_skips[i].flatten(2).permute(0,2,1) ) # B,N,D
            tmp_Qs.append( x_skips[i].flatten(2).permute(0,2,1) ) # B,N,D
            unfold_skip = self.unfolds[i](x_skips[i])
            unfold_skip = rearrange(unfold_skip, 'b (n v) f -> b f v n', n = x_skips[i].shape[1])
            rel_mat = torch.matmul(unfold_skip.detach(), unfold_skip.transpose(-1,-2).detach()).mean(-1)
            pos_v, top_p = torch.topk(-rel_mat, self.top_k[i], dim=-1, sorted=False)
            selected_skip = unfold_skip.gather(dim=-2, index=top_p.unsqueeze(-1).repeat(1,1,1,x_skips[i].shape[1]))
            tmp_skips.append( selected_skip.flatten(1,2) ) # B,N,D

        for index in range(len(self.att_blocks)):
            if index <= 5:
                new_skips.append(self.att_blocks[index](tmp_Qs[index], tmp_skips, index))
            else:
                new_skips.append(tmp_skips[index])

        for i in range(len(new_skips)):
            new_skips[i] = new_skips[i].permute(0,2,1).reshape([B]+self.layer_shapes[i][1:]) # reshape B,C,H,W

        return new_skips
    

class SparseMultiLayerAttention2(nn.Module):
    def __init__(self, layer_shapes:list):
        super().__init__()
        norms = []
        att_blocks = []
        unfold = []
        top_k = []
        ks=[(7,7),(5,5),(3,3),(1,1)]
        ss=[(7,7),(5,5),(3,3),(1,1)]
        top_k = [15,10,4,1]
        self.layer_shapes = layer_shapes
        for i in range(len(layer_shapes)):
            # norms.append(nn.LayerNorm(layer_shapes[i][1]))
            # att_blocks.append(Attention2(Q_dim=layer_shapes[i][1], KV_dim=[layer_shapes[j][1] for j in range(len(layer_shapes)) if j!=i], heads=4, dim_head=layer_shapes[i][1]))
            att_blocks.append(Attention4(Q_dim=layer_shapes[i][1], KV_dim=[layer_shapes[j][1] for j in range(len(layer_shapes)) if j!=i], heads=4, dim_head=layer_shapes[i][1]))
            unfold.append(unfoldNd.UnfoldNd(kernel_size=ks[i], stride=ss[i]))
            top_k[i] = np.ceil(np.prod(ks[i])/top_k[i]).astype(np.uint8)
            # att_blocks.append(Attention3(dim_list=[layer_shapes[j][1] for j in range(len(layer_shapes))], kv_index=i, heads=4, dim_head=layer_shapes[i][1]))

        self.att_blocks = nn.ModuleList(att_blocks)
        self.unfolds = nn.ModuleList(unfold)
        self.top_k = top_k
        # self.norms = nn.ModuleList(norms)

    def forward(self, x_skips):
        tmp_Qs =[]
        tmp_skips = []
        new_skips = []
        B = x_skips[0].shape[0]
        # 转换通道，并正则化
        for i in range(len(self.att_blocks)):
            # tmp_skips.append( x_skips[i].flatten(2).permute(0,2,1) ) # B,N,D
            tmp_Qs.append( x_skips[i].flatten(2).permute(0,2,1) ) # B,N,D
            unfold_skip = self.unfolds[i](x_skips[i])
            unfold_skip = rearrange(unfold_skip, 'b (n v) f -> b f v n', n = x_skips[i].shape[1])
            rel_mat = torch.matmul(unfold_skip.detach(), unfold_skip.transpose(-1,-2).detach()).mean(-1)
            pos_v, top_p = torch.topk(-rel_mat, self.top_k[i], dim=-1, sorted=False)
            selected_skip = unfold_skip.gather(dim=-2, index=top_p.unsqueeze(-1).repeat(1,1,1,x_skips[i].shape[1]))
            tmp_skips.append( selected_skip.flatten(1,2) ) # B,N,D

        for index in range(len(self.att_blocks)):
            if index <= 5:
                new_skips.append(self.att_blocks[index](tmp_Qs[index], tmp_skips, index))
            else:
                new_skips.append(tmp_skips[index])

        for i in range(len(new_skips)):
            new_skips[i] = new_skips[i].permute(0,2,1).reshape([B]+self.layer_shapes[i][1:]) # reshape B,C,H,W

        return new_skips


# 当前层为Q, 其他层为KV
class Attention2(nn.Module):
    def __init__(self, Q_dim, KV_dim:list, heads = 8, dim_head = 64):
        super().__init__()
        # 输入维度 dim，heads:head数量，dim_head每个头的维度
        inner_dim = dim_head *  heads #head维度
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(Q_dim, inner_dim, bias = False)
        self.q_norm = nn.LayerNorm(Q_dim)
        
        to_kvs = []
        norms = []
        for dim in KV_dim:
            to_kvs.append(nn.Linear(dim, inner_dim*2, bias = False))
            norms.append(nn.LayerNorm(dim))
        
        self.to_kvs = nn.ModuleList(to_kvs)
        self.kv_norm = nn.ModuleList(norms)

        self.to_out = nn.Linear(inner_dim * len(KV_dim), Q_dim, bias = False)

    def forward(self, x_skips, index):
        outs = []

        # Q = x_skips[index]
        # KVs = [x_skips[i] for i in range(len(x_skips)) if i!=index]

        KVs = []
        count = 0
        for i in range(len(x_skips)):
            if i==index:
                Q = self.q_norm(x_skips[index])
            else:
                KVs.append(self.kv_norm[count](x_skips[i]))
                count += 1

        q = self.to_q(Q)
        # q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), q)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        for i in range(len(self.to_kvs)):
            kv = self.to_kvs[i](KVs[i]).chunk(2, dim = -1) # 按最后一维划分变成三份
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)

            out = torch.matmul(attn, v) # B,H,N,D

            outs.append(out)

        outs = torch.concat(outs, dim=-1) # 将所有的KV结果压缩为Q的维度并输出
        outs = rearrange(outs, 'b h n d -> b n (h d)')

        return self.to_out(outs)


class Attention4(Attention2):
    def __init__(self, Q_dim, KV_dim:list, heads = 8, dim_head = 64):
        super(Attention4, self).__init__(Q_dim, KV_dim, heads, dim_head)
        
    def forward(self, query, x_skips, index):
        outs = []

        # Q = x_skips[index]
        # KVs = [x_skips[i] for i in range(len(x_skips)) if i!=index]

        KVs = []
        count = 0
        Q = self.q_norm(query)
        for i in range(len(x_skips)):
            if i!=index:
                KVs.append(self.kv_norm[count](x_skips[i]))
                count += 1

        q = self.to_q(Q)
        # q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), q)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        for i in range(len(self.to_kvs)):
            kv = self.to_kvs[i](KVs[i]).chunk(2, dim = -1) # 按最后一维划分变成三份
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)

            out = torch.matmul(attn, v) # B,H,N,D

            outs.append(out)

        outs = torch.concat(outs, dim=-1) # 将所有的KV结果压缩为Q的维度并输出
        outs = rearrange(outs, 'b h n d -> b n (h d)')

        return self.to_out(outs)

# 当前层为KV, 其他层为Q
class Attention3(nn.Module):
    def __init__(self, dim_list:list, kv_index, heads = 8, dim_head = 64):
        super().__init__()
        # 输入维度 dim，heads:head数量，dim_head每个头的维度
        # inner_dim = dim_head *  heads #head维度
        # self.heads = heads
        self.heads = []
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)

        norms = []
        to_trans = []
        concat_dim = 0
        for index, dim in enumerate(dim_list):
            norms.append(nn.LayerNorm(dim))
            if index != kv_index:
                heads = (dim_list[index] / dim).ceil() # K_dim / Q_dim = heads
                self.heads.append(heads)
                inner_dim = heads * dim_head
                to_trans.append(nn.Linear(dim, inner_dim, bias = False))
                concat_dim = (dim / dim_list[index]).ceil()*dim_head # HW -> HW/4 = 8C, HW/16 -> HW/4 == 2C
            else:
                self.heads.append(1)
                to_trans.append(nn.Linear(dim, inner_dim * 2, bias = False))

        self.to_trans = nn.ModuleList(to_trans)
        self.norms = nn.ModuleList(norms)

        self.to_out = nn.Linear(concat_dim, dim_list[kv_index], bias = False)

    def forward(self, x_skips, index):
        outs = []

        Qs = []
        for i in range(len(x_skips)):
            if i==index:
                kv = self.norms[i](x_skips[index])
                kvs= self.to_trans(kv)
                kv = kvs.chunk(2, dim = -1) # 按最后一维划分变成三份
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads[i]), kv)
            else:
                q = self.norms[i](x_skips[i])
                q = self.to_trans(q)
                q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads[i])
                Qs.append(q)

        for i in range(len(Qs)):
            dots = torch.matmul(Qs[i], k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            out = torch.matmul(attn, v) # B,H,N,D
            if i<index: # 
                # 1,HW,C -> 1, HW/4,4C
                out.append(rearrange(out, 'b h (n n1) c -> b h n (n1 c)' ,n1=4))
                # out.append(out.reshape(B, M, H*W, 4*C))
            else:
                # 16, H, W, C -> 1，4H, 4W, C
                out.append(rearrange(out, 'b (h h1) n c -> b h (h1 n) c' ,h1=4))

        outs = torch.concat(outs, dim=-1) # 将所有的KV结果压缩为Q的维度并输出
        outs = rearrange(outs, 'b h n d -> b n (h d)')

        return self.to_out(outs)