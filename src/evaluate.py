import os
import cv2
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU

from src import config as cfg
from src.utils.metrics import AverageMeter
from src.utils.losses import DeepLabv3Loss
from src.utils.data_utils import DataUtils
from src.utils.logger import set_logger_tag, logger


set_logger_tag(logger, tag="EVALUATION")


class DeepLabV3Evaluate:
    def __init__(self, args, dataset, model):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.loss_func = DeepLabv3Loss().to(args.device)

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle,
                                     num_workers=args.num_workers,
                                     pin_memory=args.pin_memory)

    def calculate_mIoU(self, mt_miou, preds, target):
        mt_miou.update(preds, target)


    def _eval(self):
        metrics = {
            'eval_loss': AverageMeter(),
            'eval_mIoU': AverageMeter()
        }
        miou_mt = MeanIoU(num_classes=self.args.num_classes, include_background=True, per_class=True)
        self.model.eval()

        for i, (images, labels, idxs) in enumerate(self.dataloader):
            with torch.no_grad():
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)

                outs = self.model(images)
                loss = self.loss_func(labels, outs)


                for j, idx in enumerate(idxs):
                    img_path, targets = self.dataset.load_voc_dataset[idx]


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=cfg['Val']['epochs'], type=int)
    parser.add_argument("--device", default=cfg['device'], type=str)
    parser.add_argument("--batch_size", default=cfg['Val']['batch_size'], type=int)
    parser.add_argument("--shuffle", default=cfg['Val']['shuffle'], type=bool)
    parser.add_argument("--num_workers", default=cfg['Val']['num_workers'], type=int)
    parser.add_argument("--pin_memory", default=cfg['Val']['pin_memory'], type=bool)
    parser.add_argument("--lr", default=cfg['Val']['lr'], type=float)
    parser.add_argument("--head_name", default=cfg['model']['head_name'], type=str)
    parser.add_argument("--backbone", default=cfg['model']['backbone'], type=str)
    parser.add_argument("--num_classes", default=cfg['Val']['dataset']['num_classes'], type=int)
    parser.add_argument("--output_stride", default=cfg['model']['output_stride'], type=int)
    parser.add_argument("--alpha", default=cfg['model']['alpha'], type=float)
    parser.add_argument("--gamma", default=cfg['model']['gamma'], type=float)

    args = parser.parse_args()
    return args