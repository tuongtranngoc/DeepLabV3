from models.backbones.resnet import resnet_feature_extraction
from heads import DeepLabHead, DeepLabHeadV3Plus

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3(nn.Module):
    def __init__(self, head_name, backbone_name, num_classes, output_stride):
        self.backbone = resnet_feature_extraction(backbone_name)
        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        else:
            aspp_dilate = [6, 12, 18]

        inplanes = 2048
        low_level_planes = 256
        
        if head_name == 'deeplabv3plus':
            self.classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        elif head_name == 'deeplabv3':
            self.classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    

