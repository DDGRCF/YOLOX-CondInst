#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self, 
                 loss_weight=1., 
                 avg_factor=1.,
                 reduction='none', 
                 loss_type="norm",
                 **kwargs):
        super(L1Loss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        assert loss_type in ['norm', 'smooth']
        self.reduction = reduction
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.avg_factor = avg_factor
        self.kwargs = kwargs
    
    def forward(self, pred, target, avg_factor=1.):
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        if self.loss_type == "norm":
            loss = F.l1_loss(pred, target, reduction="none", **self.kwargs)
        elif self.loss_type == "smooth":
            loss = F.smooth_l1_loss(pred, target, reduction="none", **self.kwargs)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss / avg_factor


class IoULoss(nn.Module):
    def __init__(self, loss_weight=1., avg_factor=1., reduction="none", loss_type="iou"):
        super(IoULoss, self).__init__()
        assert loss_type in ["iou", "giou"]
        self.reduction = reduction
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.avg_factor = avg_factor

    def forward(self, pred, target, avg_factor=1.):
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss / avg_factor

    
class DiceLoss(nn.Module):
    def __init__(self, 
                 loss_weight=1., 
                 reduction="none", 
                 loss_type='sqrt', 
                 eps=1e-5):
        super(DiceLoss, self).__init__()
        assert reduction in ["none", "sum", "mean"]
        assert loss_type in ["norm", "sqrt"]
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.eps = eps

    def forward(self, pred, target, avg_factor=1.):
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        num_pred = pred.shape[0]
        pred = pred.view(num_pred, -1)
        target = target.view(num_pred, -1)
        if self.loss_type == "sqrt":
            intersection = (pred * target).sum(1)
            union = (pred.pow(2)).sum(1) \
                + (target.pow(2)).sum(1) + self.eps
            loss = 1. - (2. * intersection / union)
        elif self.loss_type == "norm":
            intersection = 2 * (pred * target).sum(1)
            union = pred.sum(1) + target.sum(1) + self.eps
            loss = 1 - intersection / union

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return self.loss_weight * loss / avg_factor


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.,
                 reduction="none",
                 loss_type="bce_use_sigmoid",
                 **kwargs):
        super(CrossEntropyLoss, self).__init__()
        assert reduction in ["none", "mean", "sum"]
        assert loss_type in ["ce", "bce", "bce_use_sigmoid"]
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.kwargs = kwargs

    def forward(self, pred, target, avg_factor=1.):
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        if self.loss_type == "bce_use_sigmoid":
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none", **self.kwargs)
        elif self.loss_type == "bce":
            loss = F.binary_cross_entropy(pred, target, reduction="none", **self.kwargs)
        elif self.loss_type == "ce":
            loss = F.cross_entropy(pred, target, reduction="none", **self.kwargs)
        else:
            raise NotImplementedError
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return self.loss_weight * loss / avg_factor
        
        
        







