# Modified from OpenMMLab. mmdet/models/detectors/single_stage.py
# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Optional, Union

from jittordet.engine import MODELS, ConfigType, OptConfigType, load_cfg
from .single_stage import SingleStageFramework


@MODELS.register_module()
class KDSingleStageFramework(SingleStageFramework):

    def __init__(self,
                 *args,
                 teacher_config: Union[ConfigType, str, Path],
                 teacher_ckpt: Optional[str] = None,
                 eval_teacher: bool = True,
                 kd_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(teacher_config, (str, Path)):
            teacher_config = load_cfg(teacher_config)
        self.teacher = MODELS.build(teacher_config['model'])
        if teacher_ckpt is not None:
            self.teacher.load(teacher_ckpt)
        if eval_teacher:
            self.freeze(self.teacher)
        self.loss_cls_kd = MODELS.build(kd_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(kd_cfg['loss_reg_kd'])
        self.with_feat_distill = False
        if kd_cfg.get('loss_feat_kd', None):
            self.loss_feat_kd = MODELS.build(kd_cfg['loss_feat_kd'])
            self.with_feat_distill = True
        self.reused_teacher_head_idx = kd_cfg['reused_teacher_head_idx']

    @staticmethod
    def freeze(model):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
