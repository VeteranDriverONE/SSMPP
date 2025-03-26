import torch
import torch.nn as nn
import torch.nn.functional as F

""" Loss Functions -------------------------------------- """
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):  # inputs: pred,   targets: label  # 1121记：smooth=1容易把dice计算偏大，1e-5比较精确。
        # inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  # *系数 //  超参数。
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss # * 2 # 0.5 : 0.5 

        if targets.any():
            Dice_BCE = Dice_BCE * 1.1  # 加大损失。减少全黑的情况。
        return Dice_BCE


class DiceBCELoss2(nn.Module):
    def __init__(self, device, alpha=0.5, beta=0.5):
        super(DiceBCELoss2, self).__init__()
        self.device = device
        self.alpha = 0.5
        self.beta = 0.5

    def forward(self, pred, target):
        dice_loss = self.generalized_dice_loss(pred, target)
        new_target = target.max(dim=1)[1]
        bce_loss = self.BCE_loss(pred, new_target)
        Dice_BCE = self.alpha*dice_loss + self.beta*bce_loss

        if target.any():
            Dice_BCE = Dice_BCE * 1.1
        return Dice_BCE

    def generalized_dice_loss(self, pred, target,epsilon = 1e-5):
        """compute the weighted dice_loss
        Args:
            pred (tensor): prediction after softmax, shape(bath_size, channels, height, width)
            target (tensor): gt, shape(bath_size, channels, height, width)
        Returns:
            gldice_loss: loss value
        """    
        wei = torch.sum(target, axis=[0,2,3]) # (n_class,)
        wei = 1/(wei**2+epsilon)
        intersection = torch.sum(wei*torch.sum(pred * target, axis=[0,2,3]))
        union = torch.sum(wei*torch.sum(pred + target, axis=[0,2,3]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)
        return gldice_loss

    def BCE_loss(self, pred, target):
        """compute the bce_loss 
        Args:
            pred (tensor): prediction after softmax, shape(N, C, H, W)， channels = class numbers
            target (tensor): gt, shape(N, H, W), each value should be between [0,C)
        Returns:
            bce_loss : loss value
        """    
        target= target.long().to(self.device)
        bce_loss = F.cross_entropy(pred, target)
        
        return bce_loss 


# class ContrastiveLoss(nn.Module):
#     def __init__(self, batch_size, device='cuda', temperature=0.5):
#         super().__init__()
#         self.batch_size = batch_size
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
#         self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
#     def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
#         z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

#         representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
#         similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
#         sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
#         sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
#         positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
#         nominator = torch.exp(positives / self.temperature)             # 2*bs
#         denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
#         loss = torch.sum(loss_partial) / (2 * self.batch_size)
#         return loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, dim):
        # output1和output2为两个向量，label表示两向量是否是同一类，同一类为0,不同类为1
        euclidean_distance = F.pairwise_distance(output1, output2)/dim
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, dist_method='euclid'):
        super(TripletLoss, self).__init__()
        self.margin = margin     #  一般取0.3
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数
        if dist_method == 'euclid':
            self.cal_dist = self.dist_euclid
        elif dist_method == 'cos':
            self.cal_dist = self.dist_cos
        else:
            assert False, 'Not Exists method of cal dist'

    def forward(self, inputs, labels):
        n = inputs.size(0)  
        # Compute pairwise distance, replace by the official when merged
        # n = inputs.size(0)  # 获取batch_size
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n) # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
        # dist = dist + dist.t()       # 这样每个dist[i][j]代表的是第i个特征与第j个特征的平方的和
        # dist.addmm_(1, -2, inputs, inputs.t())    # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability  # 开方

        dist = self.cal_dist(inputs, n)

        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())    # 这里dist[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().view(1))       # 在i与所有有相同label的j的距离中找一个最大的
            dist_an.append(dist[i][mask[i] == 0].min().view(1))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)      # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)       # 声明一个与dist_an相同shape的全1 tensor
        y = torch.tensor(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        #prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        prec = (dist_an.data > dist_ap).data.float().mean()  #预测
        return loss, prec

    def dist_euclid(self, inputs, n):
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n) # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
        dist = dist + dist.t()       # 这样每个dist[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(1, -2, inputs, inputs.t())    # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability  #开方
        return dist
    
    def dist_cos(self, inputs, n):
        dist = 1.0 - F.cosine_similarity(inputs.unsqueeze(0), inputs.unsqueeze(1), dim=-1)
        return dist


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):  
    if b_cos:  
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)  
  
    pos_score = (view1 @ view2.T) / temperature  
    score = torch.diag(F.log_softmax(pos_score, dim=1))  
    return -score.mean()

def InfoNCE2(view1, pos, neg, temperature: float, b_cos: bool = True):  
    if b_cos:  
        view1 = F.normalize(view1, dim=1)
        pos = F.normalize(pos, dim=1)  
        neg = F.normalize(neg, dim=1)
    
    view2 = torch.concat([pos,neg])
    pos_score = (view1 @ view2.T) / temperature  
    score = F.log_softmax(pos_score, dim=1)
    return -score[:,0].mean()

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, targets):
        batch_size = features.size(0)
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        targets = targets.view(-1, 1)
        pos_mask = torch.eq(targets, targets.t()).float()
        neg_mask = 1 - pos_mask

        pos_loss = torch.exp(similarity_matrix) * pos_mask
        neg_loss = torch.exp(similarity_matrix) * neg_mask

        pos_loss = -torch.log(pos_loss.sum(1) / pos_mask.sum(1))
        neg_loss = -torch.log(neg_loss.sum(1) / neg_mask.sum(1))

        loss = pos_loss + neg_loss
        loss = loss / batch_size

        return loss.mean()

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
