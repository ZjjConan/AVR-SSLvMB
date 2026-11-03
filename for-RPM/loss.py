
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropyWithLogits(nn.Module):

    def __init__(self, reduction="mean", weighted_loss=False):
        super(BinaryCrossEntropyWithLogits, self).__init__()

        self.reduction = reduction
        self.weighted_loss = weighted_loss

 
    def forward(self, inputs, targets):
        if targets.dim() == 1:
            labels = torch.zeros_like(inputs)
            labels.scatter_(1, targets.view(-1, 1), 1.0)
        else:
            labels = targets
        weights = (1 + 6 * labels) / 7 if self.weighted_loss else None
        return F.binary_cross_entropy_with_logits(inputs, labels, weight=weights, reduction=self.reduction)


class ContrastLossWithHardNegativeMining(nn.Module):

    def __init__(self, reduction="mean", weighted_loss=False, ratio=0.5):
        super(ContrastLossWithHardNegativeMining, self).__init__()

        self.reduction = "none"
        self.ratio = ratio

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            labels = torch.zeros_like(inputs)
            labels.scatter_(1, targets.view(-1, 1), 1.0)
        else:
            labels = targets


        if inputs.requires_grad:

            loss = F.binary_cross_entropy_with_logits(inputs, labels, weight=None, reduction=self.reduction)
            pos_loss = loss[:,0].unsqueeze(-1)
            neg_loss = loss[:,1:].topk(
                int(self.ratio*(loss.shape[-1] - 1)), dim=-1
            ).values

            loss = torch.cat((pos_loss, neg_loss), dim=1).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(inputs, labels, weight=None, reduction="mean")

        return loss



class WeightedContrastLoss(nn.Module):

    def __init__(self, reduction="mean"):
        super(WeightedContrastLoss, self).__init__()

        self.reduction = reduction
 
    def forward(self, inputs, targets):
        if targets.dim() == 1:
            labels = torch.zeros_like(inputs)
            labels.scatter_(1, targets.view(-1, 1), 1.0)
        else:
            labels = targets
        weights = (1 + 9 * labels) / 10
        return F.binary_cross_entropy_with_logits(inputs, labels, weight=weights, reduction=self.reduction)