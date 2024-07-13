import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabv3Loss(nn.Module):
    def __init__(self, alpha, gamma):
        super(DeepLabv3Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CELoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, image, target):
        celoss = self.CELoss(image, target)
        floss = self.alpha * (1 - torch.exp(-celoss))**self.gamma * celoss
        return floss.mean()