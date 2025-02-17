# Modified from OpenMMLab. mmdet/models/detectors/single_stage.py
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import jittor as jt

from jittordet.engine import MODELS
from jittordet.models.utils import multi_apply, unpack_gt_instances
from jittordet.structures import InstanceList, OptInstanceList, SampleList
from jittordet.utils import reduce_mean
from .kd_stage import KDSingleStageFramework


@MODELS.register_module()
class GFLKDFramework(KDSingleStageFramework):

    def loss(self, batch_inputs: jt.Var,
             batch_data_samples: SampleList) -> Union[dict, list]:
        tea_x = self.teacher.extract_feat(batch_inputs)
        tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_crosskd_single, tea_x,
                        self.teacher.bbox_head.scales, module=self.teacher)
        stu_x = self.extract_feat(batch_inputs)
        stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_crosskd_single, stu_x,
                        self.bbox_head.scales, module=self)
        reused_cls_scores, reused_bbox_preds = multi_apply(
            self.reuse_teacher_head, tea_cls_hold, tea_reg_hold, stu_cls_hold,
            stu_reg_hold, self.teacher.bbox_head.scales)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_img_metas,
         batch_gt_instances_ignore) = outputs

        losses = self.loss_by_feat(tea_cls_scores, tea_bbox_preds, tea_x,
                                   stu_cls_scores, stu_bbox_preds, stu_x,
                                   reused_cls_scores, reused_bbox_preds,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)
        return losses

    def forward_crosskd_single(self, x, scale, module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.gfl_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred, cls_feat_hold, reg_feat_hold

    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = jt.nn.relu(reused_cls_feat)
            reused_reg_feat = jt.nn.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.gfl_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.gfl_reg(reused_reg_feat)).float()
        return reused_cls_score, reused_bbox_pred

    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()

        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdims=True)
        stu_std = stu_feat.std()
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdims=True)
        tea_std = tea_feat.std()
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def loss_by_feat(
            self,
            tea_cls_scores: List[jt.Var],
            tea_bbox_preds: List[jt.Var],
            tea_feats: List[jt.Var],
            cls_scores: List[jt.Var],
            bbox_preds: List[jt.Var],
            feats: List[jt.Var],
            reused_cls_scores: List[jt.Var],
            reused_bbox_preds: List[jt.Var],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        anchor_list, valid_flag_list = self.bbox_head.get_priors(
            featmap_sizes, batch_img_metas)

        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(jt.float32(avg_factor)).item()

        losses_cls, losses_bbox, losses_dfl,\
            new_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.bbox_head.prior_generator.strides,
                avg_factor=avg_factor)

        new_avg_factor = sum(new_avg_factor)
        new_avg_factor = reduce_mean(new_avg_factor).clamp_(min_v=1).item()
        losses_bbox = list(map(lambda x: x / new_avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / new_avg_factor, losses_dfl))
        losses = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        losses_cls_kd, losses_reg_kd, kd_avg_factor = multi_apply(
            self.pred_mimicking_loss_single,
            tea_cls_scores,
            tea_bbox_preds,
            reused_cls_scores,
            reused_bbox_preds,
            label_weights_list,
            avg_factor=avg_factor)
        kd_avg_factor = sum(kd_avg_factor)
        losses_reg_kd = list(map(lambda x: x / kd_avg_factor, losses_reg_kd))
        losses.update(
            dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd))

        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)
        return losses

    def pred_mimicking_loss_single(self, tea_cls_score, tea_bbox_pred,
                                   reused_cls_score, reused_bbox_pred,
                                   label_weights, avg_factor):
        # classification branch distillation
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # regression branch distillation
        reg_max = self.bbox_head.reg_max
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3,
                                              1).reshape(-1, reg_max + 1)
        reused_bbox_pred = reused_bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, reg_max + 1)
        reg_weights = tea_cls_score.max(dim=1).sigmoid()
        reg_weights[label_weights == 0] = 0

        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_pred,
            tea_bbox_pred,
            weight=reg_weights[:, None].expand(-1, 4).reshape(-1),
            avg_factor=4.0)

        return loss_cls_kd, loss_reg_kd, reg_weights.sum()
