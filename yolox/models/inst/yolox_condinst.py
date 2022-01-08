#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
from ..yolox import YOLOX
from ..yolo_pafpn import YOLOPAFPN
from .condinst_box_head import CondInstBoxHead


class YOLOXCondInst(YOLOX):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self, 
                 mask_head, 
                 mask_branch,
                 backbone=None, 
                 box_head=None):
        super().__init__(backbone=backbone, head=box_head)
        if backbone is None:
            backbone = YOLOPAFPN()
        if box_head is None:
            box_head = CondInstBoxHead(80)

        self.backbone = backbone
        self.box_head = box_head
        self.mask_head = mask_head
        self.mask_branch = mask_branch

    @classmethod
    def postprocess(cls, prediction, conf_thre, mask_thre, **kwargs):
        box_prediction = prediction[0]
        mask_prediction = prediction[1]
        assert box_prediction is not None

        bbox_output = [None for _ in range(len(mask_prediction))]
        mask_output = bbox_output.copy()
        score_output = bbox_output.copy()
        cls_output = bbox_output.copy()
        for i, mask in enumerate(mask_prediction):
            if not mask.size(0):
                continue
            bbox = box_prediction[i]
            conf_filter = (bbox[:, 4] * bbox[:, 5] >= conf_thre)
            bbox = bbox[conf_filter]
            mask = mask[conf_filter]
            mask = mask > mask_thre
            if not mask.size(0):
                continue

            if bbox_output[i] is None:
                bbox_output[i] = bbox[:, :4]
                score_output[i] = (bbox[:, 4] * bbox[:, 5])
                cls_output[i] = bbox[:, 6]
                mask_output[i] = mask
            else:
                bbox_output[i] = torch.cat((bbox_output[i], bbox[:, :4]))
                mask_output[i] = torch.cat((mask_output[i], mask))
                score_output[i] = torch.cat((score_output[i], bbox[:, 4] * bbox[:, 5]))
                cls_output[i] = torch.cat((cls_output[i], bbox[:, 6]))

            return bbox_output, cls_output, score_output, mask_output

    def forward(self, x, targets=None, t_masks=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, extra_infos = self.box_head(
                fpn_outs, targets, x
            )
            mask_feat = self.mask_branch(fpn_outs)
            pos_ctl_preds = extra_infos["pos_ctl_preds"]
            if pos_ctl_preds.ndim == 1:
                pos_ctl_preds = pos_ctl_preds[None]

            extra_infos["gt_masks"] = t_masks
            mask_loss = self.mask_head(mask_feat, 
                                       self.mask_branch.out_stride, 
                                       extra_infos)
            loss["total_loss"] += mask_loss["loss_mask"]
            loss.update(mask_loss)
            return loss
        else:
            extra_infos = self.box_head(fpn_outs)
            mask_feat = self.mask_branch(fpn_outs)
            mask_outs, bbox_outs = self.mask_head(mask_feat,
                                       self.mask_branch.out_stride,
                                       extra_infos)
        return mask_outs, bbox_outs
    

