import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F


from src import config as cfg
from src.utils.data_utils import DataUtils
from src.data.transformation import TransformDeepLabv3


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
    num_classes = cfg['Train']['dataset']['num_classes']
    device = cfg['device']
    C, H, W = cfg['Train']['transforms']['image_shape']
    cmap = voc_cmap()
    transform = TransformDeepLabv3()

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
            org_image = cv2.imread(img_path)
            out = model(DataUtils.to_device(image, dtype=torch.float32).unsqueeze(0))
            pred = out.max(dim=1)[1].detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            pred = cls.cmap[pred].astype(np.uint8).squeeze()
            mask = cls.cmap[mask].astype(np.uint8).squeeze()

            pred = cls.transform.decode_image(pred, org_image)
            mask = cls.transform.decode_image(mask, org_image)
            pad = 20
            H, W, C = org_image.shape
            x2mask = np.zeros((H, W * 2 + pad, C), dtype=np.uint8)
            x2mask[0:H, 0:W] = pred
            x2mask[0:H, (pad + W): (pad + W*2)] = mask

            cv2.imwrite(os.path.join(cfg['Debug'][mode.lower()], f'{i}.png'), x2mask)
    
    @classmethod
    def visualize(cls, image, result, image_path, weight=0.8, save_dir=None):
        color_map = cls.get_mask_color()
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        color_map = np.array(color_map).astype("uint8")

        vis_result = image.copy()
        for i in range(result.shape[0]):
            mask = result[i]
            c1 = np.where(mask, color_map[i, 0], vis_result[..., 0])
            c2 = np.where(mask, color_map[i, 1], vis_result[..., 1])
            c3 = np.where(mask, color_map[i, 2], vis_result[..., 2])
            pseudo_img = np.dstack((c3, c2, c1)).astype('uint8')

            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vis_result = cv2.addWeighted(vis_result, weight, pseudo_img, 1 - weight, 0)
            contour_color = (int(color_map[i, 0]), int(color_map[i, 1]), int(color_map[i, 2]))
            vis_result = cv2.drawContours(vis_result, contour, -1, contour_color, 1)

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = cfg['Debug']['prediction']
            os.makedirs(save_dir, exist_ok=True)
        image_name = os.path.split(image_path)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)


    @classmethod
    def get_mask_color(cls):
        num_classes = cls.num_classes + 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = color_map[3:]
        return color_map

