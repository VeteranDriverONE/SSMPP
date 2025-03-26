from audioop import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from collections import defaultdict

from torch.distributions.normal import Normal
from torchvision import transforms
from einops import rearrange
from einops.layers.torch import Rearrange

from MS_models.utils.metrics import ContrastiveLoss, TripletLoss, DiceBCELoss, InfoNCE, InfoNCE2, InfoNCELoss
from MS_models.ConvBlock2 import DoubleConv, normalize


# 正负原型
class ConvEncoder31(nn.Module):
    def __init__(self, inshape, in_ch=1, stage_num=4,  conv_pool=False, pool_args=None,  freeze=False):
        super(ConvEncoder31, self).__init__()
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
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args, non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args,non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args, non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args,non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                # DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, DIM=DIM),
                # DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
            kernel_size = [[3,1],[3,1],[3,1],[3,1]]
        else:
            pool_op = getattr(nn,f'MaxPool{DIM}d')
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                # DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, DIM=DIM),
                # DoubleConv(8*base_ch, 16*base_ch, 16*base_ch, DIM=DIM),
            ]
            kernel_size = [[3,1],[3,1],[3,1],[3,1]]

        self.encode_blocks = nn.ModuleList(encode_blocks[:stage_num])
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1), margin=2)

        prototypes = []
        # self.M_range = []
        self.M = [1,1,1,1]
        self.M_range = [torch.arange(0, 1.01, 1/f) for f in self.M]
        # self.M = [4,2,1,1]
        # self.M = [8,4,2,1]
        inputs = torch.rand(inshape).unsqueeze(0).unsqueeze(0)
        receptive_field = 1
        ss= 1
        self.gap = []
        self.select_index = []
        for d in range(self.stage_num):
            inputs = self.encode_blocks[d](inputs)
            
            receptive_field = receptive_field + (kernel_size[d][0]-1) * ss  # Ri = R_{i-1} * (Ki-1) * Si, Ri为当前层感受野,Si为之前卷积的步长的积（不包括当前层）
            ss = ss * kernel_size[d][1]
            gap = (receptive_field//2 * 2) // ss + 1 # 当前层不重叠感受野的间隔距离，Ri//2*2 mod ss + 1
            tmp_index = defaultdict(list)
            
            for i in range(inputs.shape[-2]):
                for j in range(inputs.shape[-1]):
                    i_mod = i % gap
                    j_mod = j % gap
                    tmp_index[f'{i_mod}-{j_mod}'].append([i, j])
            
            self.select_index.append( [torch.tensor(v) for v in tmp_index.values()] )

            prototypes.append( nn.ParameterList([nn.Parameter(torch.rand(self.M[d], inputs.shape[1]), requires_grad=False) for i in range(len(tmp_index.keys()))] ))
            # prototypes.append( nn.Parameter(torch.rand(len(tmp_index.keys()), self.M[d], inputs.shape[1])) )
        
        self.prototypes = nn.ParameterList(prototypes)
        self.seg_loss = DiceBCELoss()

    def forward1(self, x, x_mask=None):
        # X: B,C,H,W
        # x_mask, B,1,H,W
        skips =[]
        
        if x_mask is not None:
            pos_loss = torch.tensor(0)
            neg_loss = torch.tensor(0)
            contra_loss = torch.tensor(0)
            d_count = 1e-6
            for d in range(self.stage_num):
                x = self.encode_blocks[d](x)
                skips.append(x)
                x_mask = nn.functional.interpolate(x_mask, scale_factor=0.5, mode='bilinear')
                # x_mask[x_mask<0.33] = 0
                if x_mask.ceil().sum() > 0 and d <= 5:
                    d_count += 1
                    tmp_pos_loss, tmp_neg_loss = self.cal_loss(x, x_mask, d)
                    pos_loss = pos_loss + tmp_pos_loss
                    neg_loss = neg_loss + tmp_neg_loss

            return  skips, pos_loss/d_count, neg_loss/d_count, contra_loss/d_count
        
        else:
            for d in range(self.stage_num):
                x = self.encode_blocks[d](x)
                skips.append(x)
            return skips

    def forward2(self, un_x_skips, un_x_mask):
        # encoder agai
        contra_loss = torch.tensor(0)

        un_pos_loss = torch.tensor(0)
        un_neg_loss = torch.tensor(0)

        un_x_mask_round = un_x_mask.round()
        d_count = 0

        for d in range(self.stage_num):
            
            un_x_mask = nn.functional.interpolate(un_x_mask, scale_factor=0.5, mode='bilinear')
            un_x_mask_round = nn.functional.interpolate(un_x_mask_round, scale_factor=0.5, mode='bilinear')
            # un_x_mask[un_x_mask_round<0.33] = 0
            # if d <= 1 and un_x_mask_round.ceil().sum() != 0 and (1-un_x_mask_round.ceil()).sum() !=0:
            if d <= 5:
                d_count += 1
                tmp_un_pos_loss, tmp_un_neg_loss = self.cal_un_loss10(un_x_skips[d].detach(), un_x_mask, un_x_mask_round, d)
                un_pos_loss = un_pos_loss + tmp_un_pos_loss
                un_neg_loss = un_neg_loss + tmp_un_neg_loss
            
        return un_pos_loss/d_count, un_neg_loss/d_count
    
    def tsne_forward(self, x, x_mask):
        
        rep_ls = []
        rep_lab_ls = []

        for d in range(self.stage_num):
            
            x = self.encode_blocks[d](x)
            
            x_mask = nn.functional.interpolate(x_mask, scale_factor=0.5, mode='bilinear')

            rep_lab = torch.zeros_like(x_mask)

            for i in range(self.M[d]):
                rep_lab = torch.where((x_mask>self.M_range[d][i]) * (x_mask<=self.M_range[d][i+1]),
                                      torch.ones_like(x_mask)*(i+1), rep_lab)
            
            for p_iter, p in enumerate(self.select_index[d]):
                rep_lab[:,:,p[:,0],p[:,1]] += (p_iter+1)*100 + rep_lab[:,:,p[:,0],p[:,1]]

            # rep_lab[rep_lab % 100] = 100

            rep = x.flatten(2).permute(0,2,1)
            rep_lab = rep_lab.flatten(2).permute(0,2,1) 
            B,D,C = rep.shape
            rep_ls.append(rep.reshape(B*D,C))
            rep_lab_ls.append(rep_lab.reshape(B*D,1))

        return rep_ls, rep_lab_ls
    
    def cal_loss(self, x, mask, stage):
        # x1: B,C,H,W; mask1:B,1,H,W
        split_p = []
        
        pos_sim_loss = torch.tensor(0)
        neg_sim_loss = torch.tensor(0)
        p_count = 1e-12

        for p_iter, p in enumerate(self.select_index[stage]):
            patch_x = x[:,:,p[:,0],p[:,1]] #
            patch_mask = mask[:,:,p[:,0],p[:,1]]

            patch_x = patch_x.permute(0,2,1).flatten(end_dim=-2) # BHW，C
            patch_mask = patch_mask.flatten()  # BHW,1
            
            index = torch.nonzero(patch_mask)  # BHW_masked,1
            new_x = torch.index_select(patch_x, 0, index.squeeze(-1)) # BHW_masked,C
            new_mask = torch.index_select(patch_mask, 0, index.squeeze(-1)) # BHW,1)
            
            re_index = torch.nonzero(1-patch_mask.ceil())
            idx = torch.randperm(re_index.shape[0])  # 随机排序
            neg_index = re_index[idx,:].view(re_index.size()) # 排序后
            # neg_index = neg_index[:len(index)]  # 前BHW_masked个作为负样本
            neg_x = torch.index_select(patch_x, 0, neg_index.squeeze(-1))

            if len(index) == 0 or len(re_index) == 0:
                continue
            
            p_count +=1

            # 获取原型
            proto = self.prototypes[stage][p_iter] # M,C, 每个group都有一个原型

            # 正则化
            new_x = normalize(new_x)
            neg_x = normalize(neg_x)
            proto = normalize(proto)
            
            dots1 = 1 - torch.matmul(proto, new_x.transpose(-1, -2)) # M, BHW_masked, 余弦距离，越小表示该像素和原型越接近
            patch_pos_sim_loss, min_index = dots1.min(dim=0)
            pos_sim_loss = pos_sim_loss + (patch_pos_sim_loss.exp()-1).mean()

            dots2 = torch.matmul(proto, neg_x.transpose(-1, -2)) + 1 # M, BHW_unmasked， 余弦相似度，越小表示与各原型距离越远
            neg_sim_loss = neg_sim_loss + (dots2.max(dim=0)[0].exp()-1).mean()
            
            # 更新原型
            update_proto = copy.deepcopy(proto.detach())
            for i in range(self.M[stage]):
                tmp_ind = torch.where(min_index == i)[0]
                if len(tmp_ind) != 0:
                    tmp = torch.index_select(new_x, 0, tmp_ind)
                    tmp_mask = torch.index_select(new_mask, 0, tmp_ind)
                    update_proto[i,:] = tmp.mean(0)
                new_proto = (1-self.moment) * update_proto + self.moment * proto
                self.prototypes[stage][i] = new_proto.detach()
            # new_proto = (1-self.moment) * update_proto + self.moment * proto
            # self.prototypes[stage][p_iter] = new_proto.detach()

        return pos_sim_loss/p_count, neg_sim_loss/p_count

    def cal_loss2(self, x, mask, stage):
        # x1: B,C,H,W; mask1:B,1,H,W
        split_p = []
        
        pos_sim_loss = torch.tensor(0)
        neg_sim_loss = torch.tensor(0)
        p_count = 1e-12

        for p_iter, p in enumerate(self.select_index[stage]):
            patch_x = x[:,:,p[:,0],p[:,1]] #
            patch_mask = mask[:,:,p[:,0],p[:,1]]

            patch_x = patch_x.permute(0,2,1).flatten(end_dim=-2) # BHW，C
            patch_mask = patch_mask.flatten()  # BHW,1
            
            index = torch.nonzero(patch_mask)  # BHW_masked,1
            new_x = torch.index_select(patch_x, 0, index.squeeze(-1)) # BHW_masked,C
            new_mask = torch.index_select(patch_mask, 0, index.squeeze(-1)) # BHW,1)
            
            re_index = torch.nonzero(1-patch_mask.ceil())
            idx = torch.randperm(re_index.shape[0])  # 随机排序
            neg_index = re_index[idx,:].view(re_index.size()) # 排序后
            # neg_index = neg_index[:len(index)]  # 前BHW_masked个作为负样本
            neg_x = torch.index_select(patch_x, 0, neg_index.squeeze(-1))

            if len(index) == 0 or len(re_index) == 0:
                continue
            
            p_count +=1

            # 获取原型
            proto = self.prototypes[stage][p_iter] # M,C, 每个group都有一个原型

            # 正则化
            new_x = normalize(new_x)
            neg_x = normalize(neg_x)
            proto = normalize(proto)
            
            dots1 = 1 - torch.matmul(proto, new_x.transpose(-1, -2)) # M, BHW_masked, 余弦距离，越小表示该像素和原型越接近
            patch_pos_sim_loss, min_index = dots1.min(dim=0)
            pos_sim_loss = pos_sim_loss + (patch_pos_sim_loss.exp()-1).mean()

            dots2 = torch.matmul(proto, neg_x.transpose(-1, -2)) + 1 # M, BHW_unmasked， 余弦相似度，越小表示与各原型距离越远
            neg_sim_loss = neg_sim_loss + (dots2.max(dim=0)[0].exp()-1).mean()

            # 更新原型
            update_proto = copy.deepcopy(proto.detach())
            for i in range(self.M[stage]):
                tmp_ind = torch.where(min_index == i)[0]
                tmp_ind = torch.where()
                if len(tmp_ind) != 0:
                    tmp = torch.index_select(new_x, 0, tmp_ind)
                    tmp_mask = torch.index_select(new_mask, 0, tmp_ind)
                    update_proto[i,:] = tmp.mean(0)
                new_proto = (1-self.moment) * update_proto + self.moment * proto
                self.prototypes[stage][i] = new_proto.detach()

        return pos_sim_loss/p_count, neg_sim_loss/p_count

    def cal_un_loss10(self, un_x, un_mask, un_mask_round, stage):
        # 分正负， 不分patch， 加mask概率
        # x1: B,C,H,W; mask1:B,1,H,W
        # exp扩大间距
        # 损失为负

        un_loss = torch.tensor(0)
        un_pos_loss = torch.tensor(0)
        un_neg_loss = torch.tensor(0)
        
        # un_mask_flat = un_mask.flatten() # BHW
        # un_mask_round = un_mask_round.flatten().ceil() # BHW

        p_count = 1e-12
        n_count = 1e-12

        for p_iter, p in enumerate(self.select_index[stage]):
            patch_x = un_x[:,:,p[:,0],p[:,1]]
            patch_mask = un_mask[:,:,p[:,0],p[:,1]]
            patch_mask_round = un_mask_round[:,:,p[:,0],p[:,1]]

            patch_x = patch_x.permute(0,2,1).flatten(end_dim=-2) # BHW，C
            patch_mask = patch_mask.flatten()  # BHW,1
            patch_mask_round = patch_mask_round.flatten()  # BHW,1
            
            # 获取原型
            proto = self.prototypes[stage][p_iter] # M,C
            proto = normalize(proto)
            patch_x = normalize(patch_x)
            
            dots = torch.matmul(proto, patch_x.transpose(-1, -2))
            # patch_mask_ = 2 * (patch_mask-1)
            patch_mask_ = 2 * patch_mask - 1

            un_loss = un_loss - (patch_mask_ * dots.max(dim=0)[0]).mean()
            
        return un_loss/p_iter, un_neg_loss/n_count
    

class ConvEncoder53(nn.Module):
    def __init__(self, inshape, in_ch=1, stage_num=4,  conv_pool=False, pool_args=None,  freeze=False, batch_size=None):
        super(ConvEncoder53, self).__init__()
        DIM = len(inshape)
        self.mode = 'bilinear' if DIM==2 else 'trilinear'
        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96 # 0.91

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        
        if conv_pool:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args, non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args, non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args, non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args, non_line_op=nn.Tanh, non_line_op_args={}, DIM=DIM),
                # DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, DIM=DIM),
                # DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
            kernel_size = [[3,1],[3,1],[3,1],[3,1],[3,1]]
        else:
            pool_op = getattr(nn,f'MaxPool{DIM}d')
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM),
                # DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, DIM=DIM),
                # DoubleConv(8*base_ch, 16*base_ch, 16*base_ch, DIM=DIM),
            ]
            kernel_size = [[3,1],[3,1],[3,1],[3,1],[3,1]]
        encode_blocks = encode_blocks[:stage_num]
        
        encode_blocks.append(DoubleConv(8*base_ch, 16*base_ch, 16*base_ch, non_line_op=nn.Tanh, non_line_op_args={}, last_op=pool_op, last_op_args=pool_args, DIM=DIM))
        
        self.encode_blocks = nn.ModuleList(encode_blocks)
        
        prototypes = []

        self.M = [2, 2, 2, 2, 2]
        self.M_range = [[0,1],[0,1],[0,1],[0,1], [0,1]]
        # self.M_range = [torch.arange(0, 1.01, 1/f) for f in self.M]
        inputs = torch.rand(inshape).unsqueeze(0).repeat(in_ch,1,1).unsqueeze(0)
        receptive_field = 1
        ss= 1
        self.gap = []
        self.select_index = []
        for d in range(len(self.encode_blocks)):
            inputs = self.encode_blocks[d](inputs)
            
            receptive_field = receptive_field + (kernel_size[d][0]-1) * ss  # Ri = R_{i-1} * (Ki-1) * Si, Ri为当前层感受野,Si为之前卷积的步长的积（不包括当前层）
            ss = ss * kernel_size[d][1]
            gap = (receptive_field//2 * 2) // ss + 1 # 当前层不重叠感受野的间隔距离，Ri//2*2 mod ss + 1
            tmp_index = defaultdict(list)
            
            for i in range(inputs.shape[-2]):
                for j in range(inputs.shape[-1]):
                    i_mod = i % gap
                    j_mod = j % gap
                    tmp_index[f'{i_mod}-{j_mod}'].append([i, j])
            
            self.select_index.append( [torch.tensor(v) for v in tmp_index.values()] )

            prototypes.append( nn.Parameter(torch.rand(self.M[d], inputs.shape[1]), requires_grad=False) )  # False
            # prototypes.append( nn.Parameter(torch.rand(len(tmp_index.keys()), self.M[d], inputs.shape[1])) )
        
        self.prototypes = nn.ParameterList(prototypes)
        # self.seg_loss = DiceBCELoss()
        self.contra_loss = ContrastiveLoss(1)
        # self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1), margin=2)
        self.triplet_loss = TripletLoss(0.1, dist_method='cos')
        self.info_nce = InfoNCELoss(0.1)

    def forward1(self, x, x_mask=None):
        # X: B,C,H,W
        # x_mask, B,1,H,W
        skips =[]
        if x_mask is not None:
            pos_loss = torch.tensor(0)
            neg_loss = torch.tensor(0)
            contra_loss = torch.tensor(0)
            d_count = 1e-6
            for d in range(len(self.encode_blocks)):
                x = self.encode_blocks[d](x)
                if d < self.stage_num:
                    skips.append(x)

                x_mask = nn.functional.interpolate(x_mask, scale_factor=0.5, mode=self.mode)
                
                if x_mask.ceil().sum() > 0 and d <= 5:
                    d_count += 1
                    tmp_pos_loss, tmp_neg_loss, tmp_contra_loss = self.cal_loss(x, x_mask, d)
                    pos_loss = pos_loss + tmp_pos_loss
                    neg_loss = neg_loss + tmp_neg_loss
                    contra_loss = contra_loss + tmp_contra_loss

            return  skips, pos_loss/d_count, neg_loss/d_count, contra_loss/d_count
        
        else:
            for d in range(len(self.encode_blocks)):
                x = self.encode_blocks[d](x)
                skips.append(x)
            return skips

    def forward2(self, un_x_skips, un_x_mask):
        # encoder agai
        contra_loss = torch.tensor(0)
        un_pos_loss = torch.tensor(0)
        un_neg_loss = torch.tensor(0)

        un_x_mask_round = ((un_x_mask>0.5)*1).float()
        d_count = 0

        for d in range(len(self.encode_blocks)):
            
            un_x_mask = nn.functional.interpolate(un_x_mask, scale_factor=0.5, mode=self.mode)
            un_x_mask_round = nn.functional.interpolate(un_x_mask_round, scale_factor=0.5, mode=self.mode)

            if d <= 5:
                d_count += 1
                tmp_un_pos_loss, tmp_un_neg_loss = self.cal_un_loss1(un_x_skips[d].detach(), un_x_mask, un_x_mask_round, d)
                un_pos_loss = un_pos_loss + tmp_un_pos_loss
                un_neg_loss = un_neg_loss + tmp_un_neg_loss
            
        return un_pos_loss/d_count, un_neg_loss/d_count

    def tsne_forward(self, x, x_mask):
        rep_ls = []
        rep_lab_ls = []

        for d in range(len(self.encode_blocks)):
            x = self.encode_blocks[d](x)
            x_mask = nn.functional.interpolate(x_mask, scale_factor=0.5, mode='bilinear')
            rep_lab = torch.zeros_like(x_mask)

            for i in range(1, self.M[d]):
                rep_lab = torch.where((x_mask>self.M_range[d][i-1]) * (x_mask<=self.M_range[d][i]),
                                        torch.ones_like(x_mask)*(i+1), rep_lab)
            
            rep = x.flatten(2).permute(0,2,1)
            rep_lab = rep_lab.flatten(2).permute(0,2,1)
            B,D,C = rep.shape
            rep_ls.append(rep.reshape(B*D,C))
            rep_lab_ls.append(rep_lab.reshape(B*D,1))

        return rep_ls, rep_lab_ls

    def cal_loss(self, x, mask, stage):
        # x1: B,C,H,W; mask1:B,1,H,W
        
        contra_loss = torch.tensor(0)
        pos_sim_loss = torch.tensor(0)
        neg_sim_loss = torch.tensor(0)

        patch_x = x.flatten(2).permute(0,2,1).flatten(end_dim=-2) # BHW，C
        patch_mask = mask.flatten(2).permute(0,2,1).flatten()  # BHW,1
        
        index = torch.nonzero(patch_mask)  # BHW_masked,1
        new_x = torch.index_select(patch_x, 0, index.squeeze(-1)) # BHW_masked,C
        new_mask = torch.index_select(patch_mask, 0, index.squeeze(-1)) # BHW,1)
        
        neg_index = torch.nonzero(1-patch_mask.ceil())
        neg_x = torch.index_select(patch_x, 0, neg_index.squeeze(-1))

        # 获取原型
        proto = self.prototypes[stage] # M,C, 每个group都有一个原型

        # 正则化
        new_x = normalize(new_x)
        neg_x = normalize(neg_x)
        proto = normalize(proto)
        update_proto = copy.deepcopy(proto.detach())

        # dots2 = torch.matmul(proto, neg_x.transpose(-1, -2)) # M, BHW_unmasked
        
        if len(neg_index)!=0:
            update_proto[0,:] = neg_x.mean(0)
            neg_sim_loss = InfoNCE2(neg_x, proto[0:1,:], proto[1:2,:], 0.07)
        # neg_sim_loss = - (1 - 0.5*(1+dots2.max(dim=0)[0]) +1e-12).log().mean()

        if len(index) != 0:
            update_proto[1,:] = new_x.mean(0)
            pos_sim_loss = InfoNCE2(new_x, proto[1:2,:], proto[0:1,:], 0.07)
            # pos_sim_loss = pos_sim_loss - torch.index_select(dots1[i,:], 0, tmp_ind).log().mean()

        new_proto = (1-self.moment) * update_proto + self.moment * proto
        self.prototypes[stage] = new_proto.detach()

        return pos_sim_loss, neg_sim_loss, contra_loss
    

    def cal_un_loss1(self, un_x, un_mask, un_mask_round, stage):
        # 分正负， 不分patch， 加mask概率
        # x1: B,C,H,W; mask1:B,1,H,W
        # exp扩大间距
        # 损失为负

        un_loss = torch.tensor(0)
        un_pos_loss = torch.tensor(0)
        un_neg_loss = torch.tensor(0)

        patch_x = un_x.flatten(2).permute(0,2,1).flatten(end_dim=-2) # BHW，C
        patch_mask = un_mask.flatten(2).permute(0,2,1).flatten()  # BHW,1
        
        # 获取原型
        proto = self.prototypes[stage] # M,C
        proto = normalize(proto)
        patch_x = normalize(patch_x)

        dots =torch.matmul(proto, patch_x.transpose(-1, -2))
        # patch_mask_ = 2 * patch_mask - 1
        # un_loss = un_loss - (patch_mask_ * dots.max(dim=0)[0]).mean()
        max_pos, max_ind = dots.max(0)
        # un_loss = (-(1-max_ind)*(1-patch_mask + 1e-12).log() - max_ind*(patch_mask +1e-12).log()).mean()
        max_ind = max_ind * 2 - 1 
        # patch_mask = patch_mask-0.5
        patch_mask = patch_mask-0.2
        un_loss = -(max_ind * patch_mask).mean()
        
        return un_loss, un_neg_loss
    
    def cal_un_loss11(self, un_x, un_mask, un_mask_round, stage):
        # 分正负， 不分patch， 加mask概率
        # x1: B,C,H,W; mask1:B,1,H,W
        # exp扩大间距
        # 损失为负

        un_loss = torch.tensor(0)
        un_pos_loss = torch.tensor(0)
        un_neg_loss = torch.tensor(0)
        
        # un_mask_flat = un_mask.flatten() # BHW
        # un_mask_round = un_mask_round.flatten().ceil() # BHW

        un_x = un_x.permute(0,2,3,1).flatten(end_dim=-2) # BHW，C
        un_mask = un_mask.permute(0,2,3,1).flatten()  # BHW,1
        un_mask_round = un_mask_round.permute(0,2,3,1).flatten()  # BHW,1
        
        # 获取原型
        proto = self.prototypes[stage] # M,C
        proto = normalize(proto)
        un_x = normalize(un_x)
        
        dots = torch.matmul(proto, un_x.transpose(-1, -2)) + 1 #  (0-1) M,N 每个原型和每个点的相似度

        # 负样本
        tmp_ind = torch.where(un_mask_round==0)[0]
        patch_mask = torch.index_select(un_mask, 0, tmp_ind)
        if len(tmp_ind) != 0:
            un_neg_loss = ((torch.index_select(dots.max(0)[0], 0, tmp_ind)*patch_mask).log()).mean()

        tmp_count = 1e-12
        for i in range(self.M[stage]):
            tmp_ind = torch.where((un_mask_round>self.M_range[stage][i]) * (un_mask_round<=self.M_range[stage][i+1]))[0]
            if len(tmp_ind) != 0:
                tmp_count += 1
                patch_sim = torch.index_select(dots[i,:], 0, tmp_ind)
                patch_mask = torch.index_select(un_mask, 0, tmp_ind)
                un_pos_loss = un_pos_loss + ((2-patch_sim*patch_mask).log()).mean()
        un_pos_loss = un_pos_loss / tmp_count

        # patch_mask_ = 2 * (patch_mask-1)
        # patch_mask_ = 2 * patch_mask - 1
        # un_loss = un_loss - (patch_mask_ * dots.max(dim=0)[0]).mean()
            
        return un_pos_loss, un_neg_loss
    

    def cal_un_loss12(self, un_x, un_mask, un_mask_round, stage):
        # 1-P log(P) - p * log(1-P)
        un_loss = torch.tensor(0)
        un_pos_loss = torch.tensor(0)
        un_neg_loss = torch.tensor(0)
        
        # un_mask_flat = un_mask.flatten() # BHW
        # un_mask_round = un_mask_round.flatten().ceil() # BHW

        un_x = un_x.permute(0,2,3,1).flatten(end_dim=-2) # BHW，C
        un_mask = un_mask.permute(0,2,3,1).flatten()  # BHW,1
        un_mask_round = un_mask_round.permute(0,2,3,1).flatten()  # BHW,1
        
        # 获取原型
        proto = self.prototypes[stage] # M,C
        proto = normalize(proto)
        un_x = normalize(un_x)
        
        dots = torch.matmul(proto, un_x.transpose(-1, -2)) #  (-1 - 1) M,N 每个原型和每个点的相似度
        # dots_max = dots.max(0)[0]

        dots_max = 0.5 * (dots.max(0)[0] + 1) # 0 - 1
        un_pos_loss = un_mask_round.ceil() * dots_max.log() * \
                        (torch.min(un_mask, un_mask_round) + 1-un_mask_round).log() \
                        + (1-un_mask_round.ceil()) * (1-dots_max).log() * (1-un_mask+1e-6).log()
        un_pos_loss = un_pos_loss.mean()

        return un_pos_loss, un_neg_loss
    
    def cal_un_loss13(self, un_x, un_mask, un_mask_round, stage):
        # 分正负， 不分patch， 加mask概率
        # x1: B,C,H,W; mask1:B,1,H,W
        # exp扩大间距
        # 损失为负

        un_loss = torch.tensor(0)
        un_pos_loss = torch.tensor(0)
        un_neg_loss = torch.tensor(0)
        
        # un_mask_flat = un_mask.flatten() # BHW
        # un_mask_round = un_mask_round.flatten().ceil() # BHW

        un_x = un_x.permute(0,2,3,1).flatten(end_dim=-2) # BHW，C
        un_mask = un_mask.permute(0,2,3,1).flatten()  # BHW,1
        un_mask_round = un_mask_round.permute(0,2,3,1).flatten()  # BHW,1
        
        # 获取原型
        proto = self.prototypes[stage] # M,C
        proto = normalize(proto)
        un_x = normalize(un_x)
        
        dots = torch.matmul(proto, un_x.transpose(-1, -2)) #  (0-1) M,N 每个原型和每个点的相似度

        # 负样本
        tmp_ind = torch.where(un_mask_round==0)[0]
        if len(tmp_ind) != 0:
            un_neg_loss =  -torch.index_select((1-un_mask) * 0.5* (1-dots.max(0)[0]) +1e-12, 0, tmp_ind).log().mean()

        tmp_count = 1e-12
        for i in range(self.M[stage]):
            tmp_ind = torch.where((un_mask_round>self.M_range[stage][i]) * (un_mask_round<=self.M_range[stage][i+1]))[0]
            if len(tmp_ind) != 0:
                tmp_count += 1
                mid = 0.5*(self.M_range[stage][i+1] - self.M_range[stage][i]) + self.M_range[stage][i]
                un_pos_loss = un_pos_loss - torch.index_select((1 - (torch.pow(mid - un_mask,2)+1e-12).sqrt()) * 0.5*(1+dots[i,:])+1e-12, 0, tmp_ind).log().mean()

        un_pos_loss = un_pos_loss / tmp_count
        
        return un_pos_loss, un_neg_loss