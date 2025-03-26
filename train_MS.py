import torch
import torch.utils.data as Data
import numpy as np
import shutil
import time
import os

from pathlib import Path
from torchvision import utils as vutils
from MS_models.MS import MS
from MS_models.config import args
from collections import defaultdict
from MS_models.MSDdatagenerator import SemiTask012D,SemiTask022D,SemiTask072D,SemiTask082D,SemiTask083D,SemiTask092D
from MS_models.TACEdatagenerator import SemiTACESeg


def test():
    test_set  = SemiTACESeg(Path('E:\datasets\\arcade\syntax\\test'), size=args.img_shape)
    # test_set = SemiTask092D(Path('E:\\datasets\\MSD\\Task09_Spleen\\Task09\\test'), size=args.img_shape)
    # test_set = SemiTask072D(Path('E:\\datasets\\MSD\Task07_Pancreas\\Task07\\test'), size=args.img_shape)
    # test_set = SemiTask022D(Path('E:\\datasets\\MSD\\Task02_Heart\\Task02\\test'), size=args.img_shape)
    # test_set = SemiTask083D(Path('E:\\datasets\\MSD\\Task08_HepaticVessel\\Task08\\test'), size=args.img_shape)
    test_loader = Data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    ms = MS(args)
    ms = MS2(args)
    ms.load_model('MS_checkpoints\models\\best_model.pth', is_best=True)
    # ms.load_model('MS_checkpoints\models_IDVideo_0.3\\MS_1050.pt', is_best=False)

    # ms.tsne(0, test_loader, False)
    ms.test_model(test_loader=test_loader)

def train():
    args.w_seg = 1
    args.w_pos = 1
    args.w_neg = 1
    args.w_contra = 1e1 # 1e2
    args.w_proto = 1
    args.w_un_contra = 1
    args.w_un_pos = 1
    args.w_un_neg = 1

    pmar = MS(args)
    shutil.copytree('MS_models', pmar.args.save_dir, shutil.ignore_patterns(['.git', '__pycache__']))
    # pmar.load_model('MS_checkpoints\models_IDVideo4_0.3\\MS_1050.pt', is_best=False)
    
    pmar.train_model()

if __name__ == '__main__':
    # test()
    train()