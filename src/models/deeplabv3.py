from src.models.utils import IntermediateLayerGetter
from src.models.heads import DeepLabHead, DeepLabHeadV3Plus
from src.models.backbones.resnet import resnet_feature_extraction

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3(nn.Module):
    def __init__(self, head_name, backbone_name, num_classes, output_stride):
        super(DeepLabV3,self).__init__()
        backbone = resnet_feature_extraction(backbone_name)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        for i in range(1, 3):
            backbone.layer4[i].conv2.padding = (2, 2)
            backbone.layer4[i].conv2.dilation = (2, 2)

        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        else:
            aspp_dilate = [6, 12, 18]
        
        inplanes = 2048
        low_level_planes = 256
        
        if head_name.lower() == 'deeplabv3plus':
            return_layers = {
                'layer4': 'out',
                'layer1': 'low_level'
            }
            self.classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        elif head_name.lower() == 'deeplabv3':
            return_layers = {
                'layer4': 'out'
            }
            self.classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)


    def forward(self, x: torch.Tensor):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":
    model = DeepLabV3('deeplabv3plus', 'resnet101', 1000, 16)
    from src.utils.visualization import Visualizer
    Visualizer.visualize_network(model)
