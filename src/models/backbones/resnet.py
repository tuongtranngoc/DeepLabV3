import torch
import torchvision
import torch.nn as nn


model_weights = {
    'resnet50': 'ResNet50_Weights.IMAGENET1K_V1',
    'resnet101': 'ResNet101_Weights.IMAGENET1K_V1',
    'resnet152': 'ResNet152_Weights.IMAGENET1K_V1',
    'resnet50_32x4d': 'ResNeXt50_32X4D_Weights.IMAGENET1K_V1',
    'resnet101_32x8d': 'ResNeXt101_32X8D_Weights.IMAGENET1K_V1'
}

def resnet_feature_extraction(backbone_name):
    if backbone_name == 'resnet50':
        return torchvision.models.resnet50(weights=model_weights[backbone_name])
    elif backbone_name == 'resnet101':
        return torchvision.models.resnet101(weights=model_weights[backbone_name])
    elif backbone_name == 'resnet152':
        return torchvision.models.resnet152(weights=model_weights[backbone_name])
    elif backbone_name == 'resnet50_32x4d':
        return torchvision.models.resnext50_32x4d(weights=model_weights[backbone_name])
    elif backbone_name == 'resnet101_32x8d':
        return torchvision.models.resnext101_32x8d(weights=model_weights[backbone_name])
    else:
        raise Exception("The model name is not available")



if __name__ == "__main__":
    model = resnet_feature_extraction('resnet50')
    import ipdb; ipdb.set_trace();