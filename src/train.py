import argparse

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

from src.models.deeplabv3 import DeepLabV3
from src.data.pascalvoc2012 import VocDataset

set_logger_tag(logger, tag="TRAINING")


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.start_epoch = 1
        self.best_acc = 0.0
        self.create_data_loader()
        self.create_model()

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
                labels = DataUtils.to_device(labels).long()

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


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=cfg['Train']['epochs'], type=int)
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()