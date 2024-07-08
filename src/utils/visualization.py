import os
import cv2
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

from . import *


class Visualizer:
    device = cfg['device']
    C, H, W = cfg['Train']['dataset']['transforms']['image_shape']
    @classmethod
    def visualize_network(cls, model):
        os.makedirs(cfg['Debug']['model'], exist_ok=True)
        from torchview import draw_graph
        x = torch.randn(size=(cls.C, cls.H, cls.W)).to(cls.device)
        model.to(cls.device)
        draw_graph(model, input_size=x.unsqueeze(0).shape,
                   expand_nested=True,
                   save_graph=True,
                   directory=cfg['Debug']['model'],
                   graph_name=cfg['model']['backbone'])