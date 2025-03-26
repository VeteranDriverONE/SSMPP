import torch
import numpy as np
import random
import cv2
import scipy
import SimpleITK as sitk
import warnings
import math

from pathlib import Path
from torchvision.transforms import functional as F


def adjustWW(image, width=None, level=None):
    # 腹部窗宽350，窗位40

    if width is None or level is None:
        max_v = image.max()
        min_v = image.min()
        voxel_num = np.prod(image.shape)
        width = max_v
        for i in range(int(max_v), int(min_v), -1):
            if (image > i).sum() / voxel_num > 0.001:
                width = i
                break

        level = width // 2

    v_min = level - (width / 2)
    v_max = level + (width / 2)

    img = image.copy()
    img[image < v_min] = v_min
    img[image > v_max] = v_max

    img = (img - v_min) / (v_max - v_min)
    # img = (img-img.mean()) / img.std()

    return img

class SemiMSETaskBase(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        # root_path:Path, 
        # opt:describe_file, partition, size, transform, label_flag, 
        for k, v in kwargs.items():
            setattr(self,k,v)

        self.root_path = Path(self.root_path) if not isinstance(self.root_path, Path) else self.root_path
        
        image_path = self.root_path / 'image'
        label_path = self.root_path / 'label'
        
        self.images = []
        self.masks = []
        self.names = []
        self.spacings = []
        self.slices_id = []

        for img_path in image_path.glob('*.nii.gz'):
            
            itkimage = sitk.ReadImage(str(img_path))
            
            lab_path = label_path / img_path.name
            assert lab_path.exists(), '文件不存在'

            itkmask = sitk.ReadImage(str(lab_path))

            if self.size is not None:
                itkimage, itkmask = self.__resize__(self.size, itkimage, itkmask)

            image = sitk.GetArrayFromImage(itkimage)  # Z,H,W
            image = adjustWW(image, width=self.w_width, level=self.w_level)
            mask = sitk.GetArrayFromImage(itkmask)

            # mask[mask==self.label_flag] = 1
            mask = np.in1d(mask, self.label_flag).reshape(mask.shape).astype('uint8')

            self.images.append(image)
            self.masks.append(mask)
            self.names.append(img_path.name.split('.')[0])
            self.spacings.append(np.array(itkimage.GetSpacing()))
            self.slices_id.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        spacing = self.spacings[index]
        name = self.names[index]
        slice_id = self.slices_id[index]
        
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            sample = {'image':aug['image'][np.newaxis,:], 'mask':aug['mask'][np.newaxis,:],'spacing':spacing,'name':name,'slice_id':slice_id}
            return sample
        else:
            sample = {'image':image[np.newaxis,:], 'mask':mask[np.newaxis,:],'spacing':spacing,'name':name,'slice_id':slice_id}
            return sample

    def __to2d__(self):
        # 将3D换成2D
        tmp_len = len(self.images)
        
        for i in range(tmp_len):
            img_ls = [self.images[i][j] for j in range(self.images[i].shape[0])]
            msk_ls = [self.masks[i][j] for j in range(self.masks[i].shape[0])]
            # slice_id = range(len(msk_ls))

            for j in range(len(img_ls)):
                if msk_ls[j].sum() > 0:
                    self.images.append(img_ls[j])
                    self.masks.append(msk_ls[j])
                    self.names.append(self.names[i])
                    self.spacings.append(self.spacings[i])
                    self.slices_id.append(j)

        self.images = self.images[tmp_len:]
        self.masks = self.masks[tmp_len:]
        self.names = self.names[tmp_len:]
        self.spacings = self.spacings[tmp_len:]
        self.slices_id = self.slices_id[tmp_len:]

    def __partition__(self):
        # 分为有标签和无标签
        con_ls = list(zip(self.images, self.masks, self.names, self.spacings, self.slices_id))
        random.shuffle(con_ls)
        self.images, self.masks, self.names, self.spacings, self.slices_id = zip(*con_ls)

        labeled_num = round(len(self.images) * self.partition)

        self.gt_num = labeled_num
        self.un_num = len(self.images) - labeled_num

    def __crop_none_label_zone_3d__(self):
        for i in range(len(self.images)):
            ch = self.masks[i].reshape(self.masks[i].shape[0], -1).sum(axis=-1)
            pos = np.where(ch>0)[0]
            self.masks[i] = self.masks[i][pos[0]:pos[-1]+1,:,:]
            self.images[i] = self.image[i][pos[0]:pos[-1]+1,:,:]

    def __split_patch_3d__(self, size, itkimage, itkmask):
        raw_slice = self.masks.shape[0]



    def __resize__(self, new_size, img_itk, lab_itk=None):
        img_origin_size = np.array(img_itk.GetSize()) # w,h,z
        img_origin_spacing = np.array(img_itk.GetSpacing()) # z,h,w -> w,h,z

        new_size = np.array((new_size[2],new_size[1],new_size[0])) if len(new_size) == 3 and new_size[0] is not None else np.array((new_size[-1],new_size[-2],img_origin_size[2]))
        new_spacing = (img_origin_size * img_origin_spacing) / new_size

        # 图像缩放
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_itk)
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        img_itk_resample = resampler.Execute(img_itk)
        
        # img = sitk.GetArrayFromImage(img_itk_resample) # w,h,z
        # new_img = sitk.GetImageFromArray(img)
        # # new_img = sitk.Cast(sitk.RescaleIntensity(new_img), sitk.sitkUInt8)
        # new_img.SetDirection(img_itk_resample.GetDirection())
        # new_img.SetOrigin(img_itk_resample.GetOrigin())
        # new_img.SetSpacing(img_itk_resample.GetSpacing())
        if lab_itk is None:
            return img_itk_resample
        
        lab_origin_size = np.array(lab_itk.GetSize()) # w,h,z
        lab_origin_spacing = np.array(lab_itk.GetSpacing()) # z,h,w -> w,h,z
        
        assert (img_origin_size == lab_origin_size).all() \
            and (img_origin_spacing.round(4) == lab_origin_spacing.round(4)).all(), '图像和标签的origing和spacing不一致'

        # 标签缩放
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(lab_itk)
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        lab_itk_resample = resampler.Execute(lab_itk)

        # lab = sitk.GetArrayFromImage(lab_itk_resample) # w,h,z
        # new_lab = sitk.GetImageFromArray(lab)
        # new_lab.SetDirection(lab_itk_resample.GetDirection())
        # new_lab.SetOrigin(lab_itk_resample.GetOrigin())
        # new_lab.SetSpacing(lab_itk_resample.GetSpacing())

        return img_itk_resample, lab_itk_resample

class SemiTask012D(SemiMSETaskBase):
    # ct以二维图片输出
    def __init__(self, root_path:Path, describe_file=None, partition=1, size:tuple=None, transform=None, label_flag=1):
        super(SemiTask012D, self).__init__(root_path=root_path, describe_file=describe_file, partition=partition, size=size, transform=transform, label_flag=label_flag, w_width=None, w_level=None)
        self.__to2d__()
        self.__partition__()

class SemiTask022D(SemiMSETaskBase):
    # ct以二维图片输出
    def __init__(self, root_path:Path, describe_file=None, partition=1, size:tuple=None, transform=None, label_flag=1):
        super(SemiTask022D, self).__init__(root_path=root_path, describe_file=describe_file, partition=partition, size=size, transform=transform, label_flag=label_flag, w_width=None, w_level=None)
        self.__to2d__()
        self.__partition__()

class SemiTask072D(SemiMSETaskBase):
    # ct以二维图片输出
    def __init__(self, root_path:Path, describe_file=None, partition=1, size:tuple=None, transform=None, label_flag=[1,2]):
        super(SemiTask072D, self).__init__(root_path=root_path, describe_file=describe_file, partition=partition, size=size, transform=transform, label_flag=label_flag, w_width=350, w_level=40)
        self.__to2d__()
        self.__partition__()

class SemiTask082D(SemiMSETaskBase):
    # ct以二维图片输出
    def __init__(self, root_path:Path, describe_file=None, partition=1, size:tuple=None, transform=None, label_flag=[1,2]):
        super(SemiTask082D, self).__init__(root_path=root_path, describe_file=describe_file, partition=partition, size=size, transform=transform, label_flag=label_flag, w_width=350, w_level=40)
        self.__to2d__()
        self.__partition__()

class SemiTask083D(SemiMSETaskBase):
    # ct以二维图片输出
    def __init__(self, root_path:Path, describe_file=None, partition=1, size:tuple=None, transform=None, label_flag=[1,2]):
        super(SemiTask083D, self).__init__(root_path=root_path, describe_file=describe_file, partition=partition, size=size, transform=transform, label_flag=label_flag, w_width=350, w_level=40)
        self.__partition__()

class SemiTask092D(SemiMSETaskBase):
    # ct以二维图片输出
    def __init__(self, root_path:Path, describe_file=None, partition=1, size:tuple=None, transform=None, label_flag=1):
        super(SemiTask092D, self).__init__(root_path=root_path, describe_file=describe_file, partition=partition, size=size, transform=transform, label_flag=label_flag, w_width=350, w_level=40)
        self.__to2d__()
        self.__partition__()
        

class SemiHepaticVessel2d(torch.utils.data.Dataset):
    # ct以二维图片输出
    def __init__(self, root_path:Path, partition=1, size:tuple=None, transform=None):
        root_path = Path(root_path) if not isinstance(root_path, Path) else root_path

        self.transform = transform
        image_path = root_path / 'image'
        label_path = root_path / 'label'
        
        self.images = []
        self.labels = []
        self.weights = []

        for img_path in image_path.glob('*.nii.gz'):
            
            itkimage = sitk.ReadImage(str(img_path))
            image = sitk.GetArrayFromImage(itkimage) #Z,H,W
            image = adjustWW(image, width=350, level=40)

            lab_path = label_path / img_path.name
            itklabel = sitk.ReadImage(str(lab_path))
            label = sitk.GetArrayFromImage(itklabel)

            # label = (label/label.max()) if len(np.unique(label))>2 else label
            label[label>1]=1

            img_ls = [image[i] for i in range(image.shape[0])]
            lab_ls = [label[i] for i in range(label.shape[0])]

            if size is not None:
                s_h, s_w = size
                h, w = lab_ls[0].shape

                for i in range(len(img_ls)):
                    lab_ls[i] = F.resize(torch.tensor(lab_ls[i]).unsqueeze(0), size, interpolation = F.InterpolationMode.BILINEAR).squeeze().numpy()
                    img_ls[i] = F.resize(torch.tensor(img_ls[i]).unsqueeze(0), size, interpolation = F.InterpolationMode.BILINEAR).squeeze().numpy()

                # for i in range(len(img_ls)):
                #     lab_ls[i] = scipy.ndimage.zoom(lab_ls[i], (s_h/h, s_w/w), output=None, order=2, mode='constant', cval=0.0, prefilter=True)
                #     img_ls[i] = scipy.ndimage.zoom(img_ls[i], (s_h/h, s_w/w), output=None, order=2, mode='constant', cval=0.0, prefilter=True)

            for i in range(len(lab_ls)):
                lab = lab_ls[i].round().astype('uint8')
                if lab.sum() < 1:
                    continue
                
                self.images.append(img_ls[i].astype('float32'))
                self.labels.append(lab)

        con_ls = list(zip(self.images, self.labels))
        random.shuffle(con_ls)
        self.images, self.labels = zip(*con_ls)

        labeled_num = round(len(self.images) * partition)

        self.gt_num = labeled_num
        self.un_num = len(self.images) - labeled_num

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            aug = self.transform(image=image, mask=label)
            sample = {'image':aug['image'][np.newaxis,:], 'mask':aug['mask'][np.newaxis,:]}
            return sample
        else:
            sample = {'image':image[np.newaxis,:], 'mask':label[np.newaxis,:]}
            return sample
