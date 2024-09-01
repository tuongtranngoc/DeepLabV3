from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src import config as cfg


class TransformDeepLabv3(object):
    def __init__(self) -> None:
        """
        Reference: https://github.com/albumentations-team/albumentations/issues/718
        """
        self.C, self.W, self.H = cfg['Train']['transforms']['image_shape']
        self._transform = A.Compose([
            A.LongestMaxSize(max_size=max(self.W, self.H), interpolation=1),
            A.PadIfNeeded(min_height=self.H, min_width=self.W, border_mode=0, value=(0, 0, 0)),
            ToTensorV2()
        ])

    def transform(self, image, mask):
        transformed = self._transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        return transformed_image, transformed_mask

    def decode_image(self, encode_image, org_image):
        size_img = max(self.W, self.H)
        h, w = org_image.shape[:2]
        if max(h, w) > size_img:
                size_img = max(h, w) 
                encode_image = cv2.resize(encode_image, (size_img, size_img))
        
        h_pad = (size_img-h) // 2
        w_pad = (size_img-w) // 2
        out = encode_image[h_pad:h_pad+h, w_pad:w_pad+w]
        return out
    
    def augment(self, image, mask):
        H, W = image.shape[:2]
        crop_size = np.random.randint(min(H, W)//2, min(H, W))
        do_augment = A.Compose([
            A.CenterCrop(width=crop_size, height=crop_size, p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomScale(p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-15, 15), p=0.5),
        ], p=0.5)
        augmented = do_augment(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        return augmented_image, augmented_mask