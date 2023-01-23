import torch
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight):
        super(CrossEntropyLoss2d, self).__init__()
        if weight is not None:
            self.weight = weight
            self.ce_loss = nn.CrossEntropyLoss(weight)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=None)

    def forward(self, inputs_scales, targets_scales):
        inputs = inputs_scales
        targets = targets_scales
        mask = targets > 0
        targets_m = targets.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        total_loss = torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())
        return total_loss



