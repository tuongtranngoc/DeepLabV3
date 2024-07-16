import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F


from src import config as cfg


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class Visualizer:
    device = cfg['device']
    C, H, W = cfg['Train']['transforms']['image_shape']
    cmap = voc_cmap()
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
        
    @classmethod
    def save_debug(cls, image, save_dir, basename):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, basename)
        cv2.imwrite(save_path, image)

    @classmethod
    def debug_output(cls, dataset, idxs, model, mode):
        os.makedirs(cfg['Debug'][mode.lower()], exist_ok=True)
        model.eval()
        for i, idx in enumerate(idxs):
            img_path, mask_path = dataset.voc_dataset[idx]
            image, mask = dataset.get_image_mask(img_path, mask_path, False)

            out = model(image.to(cls.device).unsqueeze(0))
            pred = out.max(dim=1)[1].detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            pred = cls.cmap[pred].astype(np.uint8)
            mask = cls.cmap[mask].astype(np.uint8)
            pad = 20
            x2mask = np.zeros((cls.H, cls.W*2 + pad, cls.C), dtype=np.uint8)
            x2mask[0:cls.H, 0:cls.W] = pred
            x2mask[0:cls.H, pad + cls.W: cls.W*2] = mask

            cv2.imwrite(os.path.join(cfg['Debug'][mode.lower()], f'{i}.png'), x2mask)



