#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as  F
from ..yolo_head import YOLOXHead

class CondInstBoxHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        num_controller_params=169,
        loss_dict=dict(
            loss_iou_weight=5.0,
            loss_obj_weight=1.0,
            loss_cls_weight=1.0,
            loss_reg_weight=1.0
        ),
        nms_cfg=dict(
            pre_nms_thre=0.45,
            pre_nms_topk=1000,
            post_nms_topk=100)
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__(num_classes=num_classes,
                         width=width,
                         strides=strides,
                         in_channels=in_channels,
                         act=act,
                         depthwise=depthwise,
                         loss_dict=loss_dict)
        self.nms_cfg = nms_cfg
        self.num_controller_params = num_controller_params
        self.n_ch = 5 + num_classes + num_controller_params
        self.controller_preds = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.controller_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=num_controller_params,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def initialize_biases(self, prior_prob):
        # original init
        super().initialize_biases(prior_prob)
        # controller init
        for module in self.controller_preds:
            torch.nn.init.normal_(module.weight, std=0.01)
            torch.nn.init.constant_(module.bias, 0.)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        expanded_strides = []
        fpn_levels = []
        grids = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            ctl_output = self.controller_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, 
                    obj_output, cls_output, ctl_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )

                grids.append(grid)
                expanded_strides.append(
                   xin[0].new_full((1, grid.shape[1]), stride_this_level) 
                )
                fpn_levels.append(
                    xin[0].new_full((1, grid.shape[1]), k, dtype=torch.long)
                )
                if self.use_extra_reg:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat([
                    reg_output, obj_output.sigmoid(), 
                    cls_output.sigmoid(), 
                    ctl_output], 1
                )
                hsize, wsize = output.shape[2:]
                fpn_levels.append(
                    xin[0].new_full((1, 1, hsize, wsize), k, dtype=torch.long).view(1, -1)
                )
                expanded_strides.append(
                    xin[0].new_full((1, 1, hsize, wsize), stride_this_level).view(1, -1)
                )

                yv, xv = torch.meshgrid([torch.arange(hsize, device=output.device), 
                    torch.arange(wsize, device=output.device)])
                grids.append(torch.stack([xv, yv], 2).view(1, -1, 2))

            outputs.append(output)
        fpn_levels = torch.cat(fpn_levels, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        grids = torch.cat(grids, 1)

        if self.training:
            return self.get_losses(
                    imgs,
                    grids,
                    expanded_strides,
                    labels,
                    torch.cat(outputs, 1),
                    origin_preds,
                    fpn_levels,
                    dtype=xin[0].dtype,
                )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)

            outputs = self.decode_outputs(
                outputs, grids, expanded_strides[..., None], dtype=xin[0].type()
            )
            batch_size = outputs.shape[0]
            positions = grids * expanded_strides[..., None] \
                + 0.5 * expanded_strides[..., None]
            extra_infos = self.postprocess(outputs, 
                            fpn_levels.expand(batch_size, -1), 
                            positions.expand(batch_size, -1, -1),
                            num_classes=self.num_classes,
                            class_agnostic=True,
                            cfg=self.nms_cfg)

            return extra_infos

    def decode_outputs(self, outputs, grids, strides, dtype):
        grids = grids.type(dtype)
        strides = strides.type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    
    def get_losses(
        self, 
        imgs,
        grids,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        fpn_levels,
        dtype,
    ):
        batch_size= outputs.shape[0]
        loss_infos, extra_infos = super().get_losses(
            imgs,
            grids,
            expanded_strides,
            labels,
            outputs[..., :5+self.num_classes],
            origin_preds,
            dtype=dtype)
        pos_inds = extra_infos["pos_inds"]
        pos_ctl_preds = outputs[..., 5+self.num_classes:].view(-1, 
                            self.num_controller_params)[pos_inds]
        pos_fpn_levels = fpn_levels.repeat(batch_size, 
                    1).view(-1)[pos_inds]
        grid_positions = grids * expanded_strides[..., None] + expanded_strides[..., None] * 0.5
        pos_grid_positions = grid_positions.repeat(batch_size, 
                    1, 1).view(-1, 2)[pos_inds]

        extra_infos["pos_ctl_preds"] = pos_ctl_preds
        extra_infos["pos_fpn_levels"] = pos_fpn_levels
        extra_infos["pos_grid_positions"] = pos_grid_positions
        return loss_infos, extra_infos

    def postprocess(self, 
                    prediction, 
                    fpn_levels,
                    grids,
                    num_classes=80, 
                    class_agnostic=False,
                    cfg=None):
        
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        ctl_params = []
        pos_prediction = []
        pos_fpn_levels = []
        pos_grids = []
        img_inds = []
        num_fg = 0
        for i, image_pred in enumerate(prediction):

            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            ctl_param = image_pred[:, 5 + num_classes:]
            final_conf = image_pred[:, 4] * class_conf.squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            if len(image_pred) > cfg["pre_nms_topk"]:
                _, topk_keep_inds = final_conf.topk(cfg["pre_nms_topk"])
            else:
                topk_keep_inds = torch.arange(len(detections), device=final_conf.device)
            final_conf = final_conf[topk_keep_inds]
            detections = detections[topk_keep_inds]

            if class_agnostic:
                nms_keep_inds= torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    cfg["pre_nms_thre"]
                )
            else:
                nms_keep_inds = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    cfg["pre_nms_thre"]
                )

            final_conf = final_conf[nms_keep_inds]
            if len(nms_keep_inds) > cfg["post_nms_topk"]:
                score_thre, _ = torch.kthvalue(
                final_conf.cpu(), len(nms_keep_inds) - cfg["post_nms_topk"] + 1)
                kth_keep_inds = (final_conf >= score_thre.item()).nonzero(as_tuple=False).squeeze()
            else:
                kth_keep_inds = torch.arange(len(final_conf), device=final_conf.device)

            final_keep_inds = topk_keep_inds[nms_keep_inds[kth_keep_inds]]

            num_fg_per_img = len(final_keep_inds)
            num_fg += num_fg_per_img
            pos_prediction.append(detections[nms_keep_inds[kth_keep_inds]])
            pos_fpn_levels.append(fpn_levels[i][final_keep_inds])
            pos_grids.append(grids[i][final_keep_inds])
            ctl_params.append(ctl_param[final_keep_inds])
            img_inds.append(
                detections.new_full((num_fg_per_img, ), i, dtype=torch.long)
            )
        
        ctl_params = torch.cat(ctl_params, 0)
        img_inds = torch.cat(img_inds, 0)
        pos_prediction = torch.cat(pos_prediction, 0)
        pos_fpn_levels = torch.cat(pos_fpn_levels, 0)
        pos_grids = torch.cat(pos_grids, 0)
        outputs = dict(
            img_inds = img_inds,
            pos_ctl_preds = ctl_params,
            pos_bbox_preds = pos_prediction,
            pos_fpn_levels = pos_fpn_levels,
            pos_grid_positions = pos_grids,
            num_fg = num_fg
        )
        
        return outputs