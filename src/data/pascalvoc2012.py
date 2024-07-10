from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import os
import cv2
import glob
import json
import codecs
import xmltodict
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from collections import defaultdict

from src import config as cfg
from torch.utils.data import Dataset
from src.data.transformation import TransformDeepLabv3


class VocDataset(Dataset):
    def __init__(self, mode:str) -> None:
        self.mode = mode
        self.label_path = cfg[mode]['dataset']['anno_dir']
        self.image_path = cfg[mode]['dataset']['image_dir']
        self.ids_path = cfg[mode]['dataset']['ids_path']
        self.is_aug = cfg[mode]['transforms']['augmentation']
        self.image_shape = cfg[mode]['transforms']['image_shape']
        self.transform = TransformDeepLabv3()
        self.dataset = self.load_voc_dataset()
    
    def load_voc_dataset(self):
        dataset = []
        with codecs.open(os.path.join(self.ids_path, self.mode.lower() + '.txt'), 'r', encoding='utf-8') as f:
            data = f.readlines()
        for d in tqdm(data, desc="Loading dataset"):
            _id = d.strip()
            im_pth = os.path.join(self.image_path, _id + '.jpg')
            mask_pth = os.path.join(self.label_path, _id + '.png')
            if os.path.exists(im_pth) and os.path.exists(mask_pth):
                dataset.append([im_pth, mask_pth])

        return dataset
    
    def get_image_mask(self, img_path, mask_path, is_aug):
        image = cv2.imread(img_path)[..., ::-1]
        mask = cv2.imread(mask_path)
        if is_aug:
            image, mask = self.transform.augment(image=image, mask=mask)
        image, mask = self.transform.transform(image=image, mask=mask)
        return image, mask
    
    def __len__(self): return len(self.dataset)

    def __getitem__(self, index):
        img_path, mask_path = self.dataset[index]
        image, mask = self.get_image_mask(img_path, mask_path, is_aug=self.is_aug)
        return image, mask
