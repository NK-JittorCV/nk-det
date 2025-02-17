# Copyright (c) OpenMMLab. All rights reserved.
import jittor
import jittor.nn as nn

from jittordet.engine import MODELS
from .cross_entropy_loss import binary_cross_entropy_with_logits
from .utils import weight_reduce_loss, weighted_loss


@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       class_reduction='mean',
                                       detach_target=True):
    assert pred.size() == soft_label.size()
    target = nn.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss_func = nn.KLDivLoss(reduction='none')
    kd_loss = kd_loss_func(nn.log_softmax(pred / T, dim=1), target)

    if class_reduction == 'mean':
        kd_loss = kd_loss.mean(1)
    elif class_reduction == 'sum':
        kd_loss = kd_loss.sum(1)
    else:
        raise NotImplementedError
    kd_loss = kd_loss * (T * T)
    return kd_loss


def kd_quality_focal_loss(pred,
                          target,
                          weight=None,
                          beta=1,
                          reduction='mean',
                          avg_factor=None):
    num_classes = pred.size(1)
    if weight is not None:
        weight = weight[:, None].repeat(1, num_classes)

    target = target.detach().sigmoid()
    loss = binary_cross_entropy_with_logits(pred, target, reduction='none')
    focal_weight = jittor.abs(pred.sigmoid() - target).pow(beta)
    loss = loss * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 class_reduction='mean',
                 reduction='mean',
                 loss_weight=1.0,
                 T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.class_reduction = class_reduction
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def execute(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            class_reduction=self.class_reduction,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd


@MODELS.register_module()
class KDQualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(KDQualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss = self.loss_weight * kd_quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss
