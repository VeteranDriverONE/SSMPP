
import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
# import augmentations

# import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
from scipy import ndimage
from torch.utils.data.sampler import Sampler
# from augmentations.ctaugment import OPS
from PIL import Image
from typing import List, Tuple
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


def get_train_transforms(patch_size:Tuple[int],
                            rotation_for_DA:dict,# 旋转角度
                            mirror_axes:Tuple[int,...], # 镜像变换的轴
                            order_resampling_data: int = 3,
                            order_resampling_seg: int = 1,
                            border_val_seg:int = -1,
                            use_mask_for_norm:List[bool]= None)->AbstractTransform:
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    
    # 训练集-数据增强->旋转、缩放
    tr_transforms.append(SpatialTransform(
        patch_size_spatial,patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,
        do_scale=True,scale=(0.7,1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False,  # 随机裁剪
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))
    
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1)) # 在图像中添加高斯噪声 10%概率
    tr_transforms.append(GaussianBlurTransform((0.5,1.),different_sigma_per_channel=False,p_per_sample=0.2)) # 高斯模糊
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75,1.25),p_per_sample=0.15)) # 亮度调整
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))# 图像对比度
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5,1),per_channel=False,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes)) # 模拟低分辨率图像
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)) # 两次伽马变换，调整亮度
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))
    
    # 在指定轴上镜像翻转图像
    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes)) # x,y轴进行翻转
    
    # # 将非感兴趣区域设置为特定的值，以聚焦模型学习在特定区域的特征
    # if use_mask_for_norm is not None and any(use_mask_for_norm):
    #     tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
    #                                        mask_idx_in_seg=0,seg_outside_to=0))
    # 遍历给定数据中所有标签值，将其中的-1值改为0
    tr_transforms.append(RemoveLabelTransform(-1,0))    
    # 将seg重命名为target
    tr_transforms.append(RenameTransform('seg','target',True))
    # 将data和target转换为tensor张量
    tr_transforms.append(NumpyToTensor(['data','target'],'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms
    
def get_val_transforms():
    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1,0))
    val_transforms.append(RenameTransform('seg','target',True))
    val_transforms.append(NumpyToTensor(['data','target'],'float'))
    val_transforms = Compose(val_transforms)
    return val_transforms


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=[-2,-1])
        label = np.rot90(label, k, axes=[-2,-1])
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """
    transforms.Resize
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class TwoStreamBatchSampler_L(torch.utils.data.Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    采样：复制有标签的数据与无标签数据一致，所有无标签数据都会被读取
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, primary_batch_size):
        self.primary_indices = primary_indices # [0-16]
        self.secondary_indices = secondary_indices #[16-80]
        self.primary_batch_size = primary_batch_size  # 2
        self.secondary_batch_size = batch_size - primary_batch_size # 6
        self.part = self.secondary_batch_size // primary_batch_size # 8//2

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0
        

    def __iter__(self):
        # 返回一个batch
        primary_iter = iterate_once(self.primary_indices*(len(self.secondary_indices) // len(self.primary_indices) // self.part)) # 将1-16打乱顺序 
        secondary_iter = iterate_eternally(self.secondary_indices)
        # primary_iter = primary_iter * (len(self.secondary_indices) // len(self.primary_indices))
        return (
            primary_batch + secondary_batch for (primary_batch, secondary_batch) in zip(
                                                grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.secondary_indices) // self.secondary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)