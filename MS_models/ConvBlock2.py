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

from MS_models.utils.metrics import ContrastiveLoss, TripletLoss, DiceBCELoss, InfoNCE, InfoNCELoss


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

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



