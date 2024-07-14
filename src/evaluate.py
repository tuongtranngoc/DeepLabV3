import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config as cfg
from src.utils.metrics import AverageMeter, SegMetrics
from src.utils.losses import DeepLav3FocalLoss
from src.utils.data_utils import DataUtils
from src.utils.logger import set_logger_tag, logger


set_logger_tag(logger, tag="EVALUATION")


class DeepLabV3Evaluate:
    def __init__(self, dataset, model):
        self.args = cli()
        self.model = model
        self.dataset = dataset
        self.loss_func = DeepLav3FocalLoss(alpha=self.args.alpha, 
                                       gamma=self.args.gamma).to(self.args.device)

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=self.args.shuffle,
                                     num_workers=self.args.num_workers,
                                     pin_memory=self.args.pin_memory)
        self.mIoU_mt = SegMetrics(n_classes=self.args.num_classes)

    def _eval(self):
        metrics = {
            'eval_loss': AverageMeter(),
            'eval_mIoU': AverageMeter()
        }
        self.model.eval()
        self.mIoU_mt.reset()
        with torch.no_grad():
            for (images, labels, __) in tqdm(self.dataloader, desc="Evaluating"):
                images = DataUtils.to_device(images, dtype=torch.float32)
                labels = DataUtils.to_device(labels, dtype=torch.long)
                
                outs = self.model(images)
                loss = self.loss_func(outs, labels)
                preds = outs.max(dim=1)[1].detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()
                metrics['eval_loss'].update(loss.item())
                self.mIoU_mt.update(targets, preds)

            mIoU = self.mIoU_mt.get_results()
            metrics['eval_mIoU'].update(mIoU['Mean IoU'])
            logger.info(f'loss: {metrics["eval_loss"].get_value("mean"): .5f}, Mean IoU: {metrics["eval_mIoU"].get_value("mean"): .5f}')
        return metrics


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=cfg['device'], type=str)
    parser.add_argument("--batch_size", default=cfg['Val']['batch_size'], type=int)
    parser.add_argument("--shuffle", default=cfg['Val']['shuffle'], type=bool)
    parser.add_argument("--num_workers", default=cfg['Val']['num_workers'], type=int)
    parser.add_argument("--pin_memory", default=cfg['Val']['pin_memory'], type=bool)
    parser.add_argument("--num_classes", default=cfg['Val']['dataset']['num_classes'], type=int)
    parser.add_argument("--alpha", default=cfg['model']['alpha'], type=float)
    parser.add_argument("--gamma", default=cfg['model']['gamma'], type=float)

    args = parser.parse_args()
    return args