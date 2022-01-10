#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from ..yolox_base import Exp
from yolox.core import InstTrainer

class CondInstExp(Exp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        self.act = 'silu'
        # box head
        self.box_loss_dict = dict(
            loss_iou_weight=5.0,
            loss_obj_weight=1.0,
            loss_cls_weight=1.0,
            loss_reg_weight=1.0
        )
        # dynamic mask head
        self.mask_stride_out = 4
        self.size_of_interest = [64, 128, 256]
        self.dynamic_mask_loss_dict = dict(
            loss_mask_weight=5.0
        )

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --------------- transform config ----------------- #
        self.trainer = InstTrainer

        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mixup = True
        self.with_mask = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 0
        self.max_epoch = 24
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 10
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.no_validate = True
        self.save_metric = "segm"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.metric=["bbox", "segm"]
        self.postprocess_cfg = dict(
            # for box out
            pre_nms_thre=0.65,
            pre_nms_topk=1000,
            post_nms_topk=100,
            # for mask out
            conf_thre=0.01,
            mask_thre=0.5
        )
        self.test_size = (640, 640)
        # -----------------  demo config ------------------ #
        self.vis_conf_thre=0.3

    def get_model(self):
        from yolox.models import (YOLOXCondInst, CondInstPAFPN, 
                                  CondInstBoxHead, CondInstMaskBranch,
                                  DynamicMaskHead)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = CondInstPAFPN(self.depth, self.width, 
                        in_channels=in_channels, 
                        act=self.act)
            box_head = CondInstBoxHead(self.num_classes, self.width, 
                                       in_channels=backbone.out_channels, 
                                       loss_dict=self.box_loss_dict, act=self.act,
                                       nms_cfg=self.postprocess_cfg)
            mask_branch = CondInstMaskBranch(in_channels=backbone.out_channels, 
                                        width=self.width, out_stride=box_head.strides[0])
            mask_head = DynamicMaskHead(in_channel=mask_branch.out_channel, 
                                        mask_stride_out=self.mask_stride_out,
                                        soi=self.size_of_interest, act=self.act, 
                                        loss_dict=self.dynamic_mask_loss_dict)

            self.model = YOLOXCondInst(mask_head, 
                                       mask_branch,
                                       backbone=backbone, 
                                       box_head=box_head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        # self.model.mask_head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
            mask_collate
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
                with_mask=self.with_mask
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            with_mask=self.with_mask
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        dataloader_kwargs["collate_fn"] = mask_collate

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

        
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            pg0_l, pg1_l, pg2_l = [] , [], []

            for k, v in self.model.named_modules():
                if "backbone" in k:
                    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                        pg2_l.append(v.bias)  # biases
                        # v.bias.requires_grad = False
                    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                        pg0_l.append(v.weight)  # no decay
                        # v.weight.requires_grad = False
                    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                        pg1_l.append(v.weight)  # apply decay
                        # v.weight.requires_grad = False
                elif "head" in k:
                    if "controller_preds" not in k:
                        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                            pg2_l.append(v.bias)  # biases
                        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                            pg0_l.append(v.weight)  # no decay
                        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                            pg1_l.append(v.weight)  # apply decay
                    else:
                        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                            pg2.append(v.bias)  # biases
                        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                            pg0.append(v.weight)  # no decay
                        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                            pg1.append(v.weight)  # apply decay
                else:
                    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                        pg2.append(v.bias)  # biases
                    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                        pg0.append(v.weight)  # no decay
                    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                        pg1.append(v.weight)  # apply decay
            # norm learning rate
            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay},
            )  # add pg1 with weight_decay

            optimizer.add_param_group({"params": pg2})
            # low learning rate
            optimizer.add_param_group(
                {"params": pg1_l, "weight_decay": self.weight_decay, "lr": lr * 0.01},
            )
            optimizer.add_param_group({"params": pg0_l, "lr": lr * 0.01})
            optimizer.add_param_group({"params": pg2_l, "lr": lr * 0.01})
            self.optimizer = optimizer

        return self.optimizer