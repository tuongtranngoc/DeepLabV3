import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config as cfg

from src.utils.visualization import Visualizer
from src.utils.losses import DeepLav3FocalLoss
from src.utils.tensorboard import Tensorboard
from src.utils.data_utils import DataUtils
from src.utils.metrics import AverageMeter
from src.evaluate import DeepLabV3Evaluate
from src.utils.schedulers import PolyLR

from src.models.heads import convert_to_separable_conv
from src.utils.logger import logger, set_logger_tag
from src.models.utils import set_bn_momentum
from src.data.pascalvoc2012 import VocDataset
from src.models.deeplabv3 import DeepLabV3

set_logger_tag(logger, tag="TRAINING")


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.start_iter = 0
        self.start_epoch = 0
        self.best_mIoU = 0.0
        self.create_data_loader()
        self.create_model()
        self.evaluate = DeepLabV3Evaluate(self.valid_dataset, self.model)

    def create_data_loader(self):
        self.train_dataset = VocDataset(mode='Train')
        self.valid_dataset = VocDataset(mode='Val')
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory)
    
    def create_model(self):
        self.model = DeepLabV3(head_name=self.args.head_name, 
                               backbone_name=self.args.backbone,
                               num_classes=self.args.num_classes,
                               output_stride=self.args.output_stride)
        
        set_bn_momentum(self.model.backbone, momentum=0.01)
        convert_to_separable_conv(self.model.classifier)
        self.model.to(self.args.device)
        self.loss_func = DeepLav3FocalLoss(alpha=self.args.alpha, gamma=self.args.gamma).to(self.args.device)
        self.optimizer = torch.optim.SGD(params=[
                {'params': self.model.backbone.parameters(), 'lr': self.args.lr},
                {'params': self.model.classifier.parameters(), 'lr': self.args.lr},
            ], lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        
        self.scheduler = PolyLR(self.optimizer, max_iters=self.args.total_itrs, power=0.1)

    def train(self):
        while self.start_iter <= self.args.total_itrs:
            mt_loss = AverageMeter()
            self.start_epoch += 1
            for bz, (images, labels, __) in enumerate(self.train_loader):
                self.model.train()
                self.start_iter += 1
                images = DataUtils.to_device(images, torch.float32)
                labels = DataUtils.to_device(labels, torch.long)
                self.optimizer.zero_grad()
                outs = self.model(images)

                loss = self.loss_func(outs, labels)

                loss.backward()
                self.optimizer.step()
                mt_loss.update(loss.item())

                print(f"Epoch {self.start_epoch} - batch {bz+1}/{len(self.train_loader)}, loss: {mt_loss.get_value(): .5f}", end='\r')

                Tensorboard.add_scalars('train_loss', self.start_iter, loss=mt_loss.get_value('mean'))

                if self.start_iter % self.args.eval_step == 0:
                    metrics = self.evaluate._eval()
                    Tensorboard.add_scalars("eval_loss", self.start_iter, loss=metrics['eval_loss'].get_value('mean'))
                    Tensorboard.add_scalars("eval_meanIoU", self.start_iter, mIoU=metrics['eval_mIoU'].get_value('mean'))

                    # Save best checkpoint
                    ckpt_dir = os.path.join(cfg['Debug']['ckpt_dirpath'], self.args.backbone.lower() + '_' + self.args.head_name.lower())
                    current_mIoU = metrics['eval_mIoU'].get_value('mean')
                    if current_mIoU > self.best_mIoU:
                        self.best_mIoU = current_mIoU
                        best_ckpt_pth = os.path.join(ckpt_dir, 'best.pt')
                        self.save_ckpt(best_ckpt_pth, self.best_mIoU, self.start_iter)

                    # Save last checkpoint
                    last_ckpt_path = os.path.join(ckpt_dir, 'last.pt')
                    self.save_ckpt(last_ckpt_path, self.best_mIoU, self.start_iter)

                # Debug after each training epoch
                if self.args.debug_mode:
                    Visualizer.debug_output(self.train_dataset, cfg['Debug']['debug_idxs'], self.model, mode='Train')
                    Visualizer.debug_output(self.valid_dataset, cfg['Debug']['debug_idxs'], self.model, mode='Val')

                self.scheduler.step()

    def save_ckpt(self, save_path, best_mIoU, cur_iter):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_mIoU": best_mIoU,
            "cur_iter": cur_iter,
        }
        logger.info(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)
    
    def resume_training(self, ckpt):
        self.best_mIoU = ckpt['best_mIoU']
        cur_iter = ckpt['cur_iter'] + 1
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.model.load_state_dict(ckpt['model'])

        return cur_iter


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_itrs", default=cfg['Train']['total_itrs'], type=int)
    parser.add_argument("--eval_step", default=cfg['Val']['eval_step'], type=int)
    parser.add_argument("--device", default=cfg['device'], type=str)
    parser.add_argument("--batch_size", default=cfg['Train']['batch_size'], type=int)
    parser.add_argument("--shuffle", default=cfg['Train']['shuffle'], type=bool)
    parser.add_argument("--num_workers", default=cfg['Train']['num_workers'], type=int)
    parser.add_argument("--pin_memory", default=cfg['Train']['pin_memory'], type=bool)
    parser.add_argument("--lr", default=cfg['Train']['lr'], type=float)
    parser.add_argument("--head_name", default=cfg['model']['head_name'], type=str)
    parser.add_argument("--backbone", default=cfg['model']['backbone'], type=str)
    parser.add_argument("--num_classes", default=cfg['Train']['dataset']['num_classes'], type=int)
    parser.add_argument("--output_stride", default=cfg['model']['output_stride'], type=int)
    parser.add_argument("--alpha", default=cfg['model']['alpha'], type=float)
    parser.add_argument("--gamma", default=cfg['model']['gamma'], type=float)
    parser.add_argument("--debug_mode", default=cfg['Debug']['debug_mode'], type=bool)
    parser.add_argument("--weight_decay", default=cfg['Train']['weight_decay'], type=float)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()