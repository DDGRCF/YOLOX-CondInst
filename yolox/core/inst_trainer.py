#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import time

import torch
from .trainer import Trainer

class InstTrainer(Trainer):
    def __init__(self, exp, args):
        super().__init__(exp, args)
        self.save_metric = exp.save_metric

    def before_train(self):
        super().before_train()
        self.lr_scheduler.cur_lr = self.lr_scheduler.lr

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets, t_masks = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, 
                targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, t_masks)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        # TODO; update learning rate
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        adjust_rate = lr / self.lr_scheduler.cur_lr
        self.lr_scheduler.cur_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = adjust_rate * param_group["lr"]

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )
