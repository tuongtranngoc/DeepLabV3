import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config as cfg

from src.utils.data_utils import DataUtils
from src.utils.visualization import Visualizer
from src.utils.logger import logger, set_logger_tag
from src.utils.losses import DeepLabv3Loss
from src.utils.metrics import AverageMeter
from src.utils.tensorboard import Tensorboard
from src.evaluate import DeepLabV3Evaluate

from src.models.deeplabv3 import DeepLabV3
from src.data.pascalvoc2012 import VocDataset

set_logger_tag(logger, tag="TRAINING")


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.start_epoch = 1
        self.best_mIoU = 0.0
        self.create_data_loader()
        self.create_model()
        self.evaluate = DeepLabV3Evaluate(args, self.valid_dataset, self.model)

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
                               output_stride=self.args.output_stride).to(self.args.device)
        self.loss_func = DeepLabv3Loss(alpha=self.args.alpha, gamma=self.args.gamma).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            mt_loss = AverageMeter()

            for bz, (images, labels, __) in enumerate(self.train_loader):
                self.model.train()
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels.long())

                outs = self.model(images)

                loss = self.loss_func(outs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mt_loss.update(loss.item())

                print(f"Epoch {epoch} - batch {bz+1}/{len(self.train_loader)}, loss: {mt_loss.get_value(): .5f}", end='\r')

                Tensorboard.add_scalars('train_loss', epoch,
                                        loss=mt_loss.get_value('mean'))

            logger.info(f"Epoch: {epoch} - loss: {mt_loss.get_value('mean'): .5f}")

            if epoch % self.args.eval_step == 0:
                metrics = self.evaluate._eval()
                Tensorboard.add_scalars("eval_loss", epoch,
                                        loss=metrics['eval_loss'].get_value('mean'))
                Tensorboard.add_scalars("eval_meanIoU", epoch,
                                        mIoU=metrics['eval_mIoU'].get_value('mean'))

                # Save best checkpoint
                current_mIoU = metrics['eval_mIoU'].get_value('mean')
                if current_mIoU > self.best_mIoU:
                    self.best_mIoU = current_mIoU
                    best_ckpt_pth = os.path.join(cfg['Debug']['ckpt_dirpath'], self.args.backbone + '_' + self.args.head_name, 'best.pt')
                    self.save_ckpt(best_ckpt_pth, self.best_mIoU, epoch)

            # Save last checkpoint
            last_ckpt_path = os.path.join(cfg['Debug']['ckpt_dirpath'], self.args.backbone  + '_' + self.args.head_name, 'last.pt')
            self.save_ckpt(last_ckpt_path, self.best_mIoU, epoch)

            # Debug after each training epoch
            if self.args.debug_mode:
                pass


    def save_ckpt(self, save_path, best_mIoU, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_mIoU": best_mIoU,
            "epoch": epoch,
        }
        logger.info(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)

    def resume_training(self, ckpt):
        self.best_mIoU = ckpt['best_mIoU']
        start_epoch = ckpt['epoch'] + 1
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.model.load_state_dict(ckpt['model'])

        return start_epoch



def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=cfg['Train']['epochs'], type=int)
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()