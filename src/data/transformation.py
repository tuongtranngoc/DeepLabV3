from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


from src import config as cfg


class TransformDeepLabv3(object):
    def __init__(self) -> None:
        self.image_shape = cfg['Train']['transforms']['image_shape']
        self._transform = A.Compose([
            A.Resize(self.image_shape[1], self.image_shape[2]),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])
        self._augment = A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.VerticalFlip(p=0.5),
            A.MedianBlur(p=0.1, blur_limit=5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        ], p=0.5)

    def transform(self, image, mask):
        transformed = self._transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        return transformed_image, transformed_mask
    
    def augment(self, image, mask):
        augmented = self._augment(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        return augmented_image, augmented_mask