import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabv3Loss(nn.Module):
    def __init__(self, alpha, gamma, ignore_index=255):
        super(DeepLabv3Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        celoss = F.cross_entropy(preds, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-celoss).to(preds.device)
        floss = self.alpha * (1 - pt)**self.gamma * celoss
        return floss.mean()