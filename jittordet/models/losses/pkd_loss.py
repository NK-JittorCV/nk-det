# Copyright (c) OpenMMLab. All rights reserved.
import jittor.nn as nn

from jittordet.engine import MODELS
from .utils import weighted_loss


def norm(feat):
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdims=True)
    std = feat.std()
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


@weighted_loss
def pkd_loss(pred, target):
    pred = norm(pred)
    target = norm(target)
    return (pred - target).sqr() / 2


@MODELS.register_module()
class PKDLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PKDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * pkd_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
