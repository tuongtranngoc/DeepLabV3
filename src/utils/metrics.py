from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from . import cfg

import torch
import numpy as np
from torchmetrics.segmentation.mean_iou import MeanIoU


class AverageMeter(object):
    """Calculate average/sum value after each time
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.value = 0
        self.count = 0
    
    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        
    def get_value(self, summary_type=None):
        if summary_type == 'mean':
            return self.avg
        elif summary_type == 'sum':
            return self.sum
        else:
           return self.value


class MIoUTorchMetric(object):
    def __init__(self, num_classes):
        self.miou_mt = MeanIoU(num_classes=num_classes, include_background=True, per_class=True)
    
    def compute_acc(self, preds, target):
        """MEAN INTERSECTION OVER UNION (MIOU)
        Reference: https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html
        """
        self.miou_mt.update(preds, target)