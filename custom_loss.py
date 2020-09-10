import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean',weight=None,pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduction
        self.weights=weight
        self.pos_weights=pos_weight

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none',weight=self.weights,pos_weight=self.pos_weights)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none',weight=self.weights)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce=='none':
            return F_loss
        elif self.reduce=='sum':
            return torch.sum(F_loss)
        else:
            return torch.mean(F_loss)