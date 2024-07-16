import os
import cv2
import glob
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from src import config as cfg
from src.models.deeplabv3 import DeepLabV3
from src.utils.data_utils import DataUtils
from src.data.pascalvoc2012 import VocDataset
from src.data.transformation import TransformDeepLabv3
from src.utils.logger import set_logger_tag, logger

set_logger_tag(logger, tag="PREDICTING")

class Predictor:
    def __init__(self, args):
        self.args = args
        self.model = DeepLabV3(head_name=self.args.head_name,
                               backbone_name=self.args.backbone,
                               num_classes=self.args.num_classes,
                               output_stride=self.args.out_stride)
        ckpt_path = os.path.join(cfg["Debug"]["ckpt_dirpath"], 
                                 self.args.backbone.lower() + "_" +
                                 self.args.head_name.lower(),
                                 "best.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.args.device)['model'])
        else:
            logger.warning(f"Not exist checkpoint path")
        
        self.model.to(self.args.device)
        self.transfom = TransformDeepLabv3()

    def predict(self, image_path):
        image = cv2.imread(image_path)
        transformed_img = self.transfom._transform(image=image[..., ::-1])['image']
        transformed_img = transformed_img.to(self.args.device)
        mask = self.model(transformed_img)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--device", default=cfg["device"], type=str)
    parser.add_argument("--head_name", default=cfg["model"]["head_name"], type=str)
    parser.add_argument("--backbone", default=cfg["model"]["backbone"], type=str)
    parser.add_argument("--num_classes", default=cfg["Train"]["dataset"]["num_classes"], type=int)
    parser.add_argument("--out_stride", default=cfg["model"]["output_stride"], type=int)
    parser.add_argument("--out_dir", default=cfg["Debug"]["prediction"], type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    predictor = Predictor(args)

    VAL_DATASET = VocDataset(mode='Val')
    dataset = VAL_DATASET.voc_dataset

    for idx in tqdm(range(100)):
        image_path = dataset[idx][0]
        res = predictor.predict(image_path)