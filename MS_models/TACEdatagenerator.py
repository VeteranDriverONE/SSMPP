import json
import cv2
import random
import numpy as np
import torch

from pycocotools.coco import COCO
from collections import defaultdict
from pathlib import Path
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


class TACESeg(torch.utils.data.Dataset):
    def __init__(self, root_path:Path, size=None, transform=None, multi_seg=False):
        # multi_seg=False:二类分割，multi_seg=True:多类分割
        self.new_size = size  # H,W
        self.transform = transform

        images_root = root_path / 'images'
        anno_file = list((root_path / 'annotations').glob('*.json'))
        assert len(anno_file)==1, 'json文件不存在或过多'
        anno_file = anno_file[0]
        with open(anno_file) as f:
            json_info = json.load(f)
        image_info = json_info["images"]
        anno_info = json_info["annotations"]
        
        self.images = []
        self.masks = []
        self.bboxs = []
        self.names = []

        image_dict = {}
        anno_dict = defaultdict(list)

        coco = COCO(anno_file)

        for tmp_info in image_info:
            image_dict[tmp_info["id"]] = tmp_info
        
        for tmp_info in anno_info:
            anno_dict[tmp_info["image_id"]].append(tmp_info)

        for img_id in image_dict.keys():
            if len(anno_dict[img_id]) == 0:
                continue
            img_filename = image_dict[img_id]['file_name']
            img_path = images_root / img_filename
            image = cv2.imread(str(img_path), 0)

            mask = np.zeros(image.shape, np.int32)[np.newaxis,...]
            mask = np.repeat(mask, 26, 0)

            for anno in anno_dict[img_id]:
                class_id = anno['category_id']
                bbox = np.array(anno['bbox'])
                points = np.array([anno["segmentation"][0][::2], anno["segmentation"][0][1::2]], np.int32).T
                points = points.reshape((-1, 1, 2))
                tmp = np.zeros(image.shape, np.int32)
                cv2.fillPoly(tmp, [points], (1))
                mask[class_id] += tmp
                mask[class_id, mask[class_id]>0] = 1

            # mask = np.zeros_like(image)
            # for anno in anno_dict[img_id]:
            #     class_id = anno['category_id']
            #     bbox = np.array(anno['bbox'])

            #     # for segment in anno['segmentation']:
            #     #     cv2.fillPoly(mask, [np.array(segment, dtype=np.int32)], (255))
                
            #     tmp_mask = coco.annToMask(anno) * class_id
            #     mask = np.where(mask, tmp_mask>0, tmp_mask)

            if not multi_seg:
                mask = mask.sum(0, keepdims=True)
                mask[mask>0] = 1
                # tmp = np.zeros_like(mask)
                # mask = np.concatenate([tmp, mask], axis=0)

            if size is not None:
                bbox = bbox * np.repeat( size / np.array(image.shape), 2)
                image = self.__resize__(np.transpose(image, [1,0]), size) # cv2resize 接受WH
                image = np.transpose(image, [1,0])  # H,W
                mask = self.__resize__(np.transpose(np.uint8(mask), [2,1,0]), size)
                mask = mask[..., np.newaxis] if len(mask.shape)==2 else mask
                mask = np.transpose(mask, [2,1,0])
            
            self.images.append(image)
            self.masks.append(mask)
            self.bboxs.append(bbox)
            self.names.append(str(img_id))

        print(f"Dataset loaded. Total nummber of image:{len(image_info)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        bbox = self.bboxs[index]
        name = self.names[index]
        
        if self.transform:
            aug = self.transform(image=image, mask=mask[0])
            sample = {'image':aug['image'][np.newaxis,:], 'mask':aug['mask'][np.newaxis,:], 'name':name}
            return sample
        else:
            sample = {'image':image[np.newaxis,:], 'mask':mask, 'bbox':bbox, 'name':name}
            return sample

    def __partition__(self, partition):
        # 分为有标签和无标签
        con_ls = list(zip(self.images, self.masks, self.bboxs, self.names))
        random.shuffle(con_ls)
        self.images, self.masks, self.bboxs, self.names = zip(*con_ls)

        labeled_num = round(len(self.images) * partition)

        self.gt_num = labeled_num
        self.un_num = len(self.images) - labeled_num
        print(f"The partition of ground truth  is:{partition}, gt_num:{self.gt_num}, un_num:{self.un_num}")

    def __resize__(self, img, new_size):
        return cv2.resize(img, new_size, interpolation=1)



class SemiTACESeg(TACESeg):
    def __init__(self, root_path:Path, partition=0.3, size:tuple=None, transform=None, multi_seg=False):
        super(SemiTACESeg, self).__init__(root_path=root_path, size=size, transform=transform, multi_seg=multi_seg)
        self.__partition__(partition=partition)


if __name__ == '__main__':
    taceseg = TACESeg(Path("E:\datasets\\arcade\stenosis\\train"))