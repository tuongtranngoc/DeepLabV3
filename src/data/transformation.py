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
            A.Normalize(),
            ToTensorV2()
        ])

    def transform(self, image, mask):
        transformed = self._transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        return transformed_image, transformed_mask
    
    def augment(self, image, mask):
        H, W = image.shape[:2]
        crop_size = np.random.randint(min(H, W)//2, min(H, W))
        do_augment = A.Compose([
            A.CenterCrop(width=crop_size, height=crop_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-15, 15), p=0.5),
        ], p=0.5)
        augmented = do_augment(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        return augmented_image, augmented_mask