import torch
import torchvision
import torch.nn as nn


model_weights = {
    'mobilenet_v3_small': 'MobileNet_V3_Small_Weights.IMAGENET1K_V1',
    'mobilenet_v3_large': 'MobileNet_V3_Large_Weights.IMAGENET1K_V1',
}

def resnet_feature_extraction(backbone_name):
    if backbone_name == 'mobilenet_v3_small':
        return torchvision.models.mobilenet_v3_small(weights=model_weights[backbone_name])
    elif backbone_name == 'mobilenet_v3_large':
        return torchvision.models.mobilenet_v3_large(weights=model_weights[backbone_name])
    else:
        raise Exception("The model name is not available")