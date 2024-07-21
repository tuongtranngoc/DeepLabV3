from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from typing import Any, Literal

import torch
import numpy as np
from torch import Tensor
from torchmetrics import Metric


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
        

def _compute_intersection_and_union(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = 255,
    include_background: bool = False,
    input_format: Literal["one-hot", "index", "predictions"] = "index",
) -> tuple[Tensor, Tensor]:
    ignore_idxs = torch.where(target==ignore_index)
    target[ignore_idxs] = 0
    preds[ignore_idxs] = 0
    if input_format in ["index", "predictions"]:
        if input_format == "predictions":
            preds = preds.argmax(1)
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes)

    if not include_background:
        preds[..., 0] = 0
        target[..., 0] = 0
    
    reduce_axis = list(range(1, preds.ndim - 1))
    intersection = torch.sum(torch.logical_and(preds, target), dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)
    union = target_sum + pred_sum - intersection

    return intersection, union


class MeanIoU(Metric):
    """Fix bug: Segmentation metric
    Reference: https://github.com/Lightning-AI/torchmetrics/issues/2558
    """
    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        input_format: Literal["one-hot", "index", "predictions"] = "index",
        **kwargs: Any,
    ) -> None:
        Metric.__init__(self, **kwargs)

        self.num_classes = num_classes
        self.include_background = include_background
        self.per_class = per_class
        self.input_format = input_format

        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        intersection, union = _compute_intersection_and_union(
            preds, target, self.num_classes, self.include_background, self.input_format
        )
        self.intersection += intersection.sum(0)
        self.union += union.sum(0)
    
    def compute(self) -> Tensor:
        iou_valid = torch.gt(self.union, 0)

        iou = torch.where(
            iou_valid,
            torch.divide(self.intersection, self.union),
            torch.nan,
        )

        if self.per_class:
            return iou
        else:
            return torch.mean(iou[iou_valid])