#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from ..network_blocks import BaseConv
from ..yolo_pafpn import YOLOPAFPN


class CondInstPAFPN(YOLOPAFPN):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        out_strides=[8, 16, 32],
        depthwise=False,
        act="silu",
        num_outs=3,
    ):
        super().__init__(depth=depth,
                         width=width, 
                         in_features=in_features, 
                         in_channels=in_channels, 
                         depthwise=depthwise,
                         act=act)
        
        self.num_outs = num_outs
        self.out_strides = out_strides
        out_channels = in_channels.copy()
        if num_outs > len(in_channels):
            extra_layers_in_channels = [in_channels[-1]] * int(num_outs - len(in_channels)) 
            self.extra_layer_list = nn.ModuleList()
            for i, c in enumerate(extra_layers_in_channels):  
                self.out_strides.append(out_strides[-1] // 2)
                if i == len(extra_layers_in_channels) - 1:
                    self.extra_layer_list.append(
                        BaseConv(int(c * width), int(c * width), 3, 2, 1, 
                                        with_act=False, with_bn=True)
                    )
                else:
                    self.extra_layer_list.append(
                        BaseConv(int(c * width), int(c * width), 3, 2, 1, 
                                        with_act=True, with_bn=True)
                    )
                    
            out_channels.extend(extra_layers_in_channels)
        self.out_channels = out_channels


    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        if self.num_outs > len(self.in_channels):
            for layer_module in self.extra_layer_list:
                outputs.append(layer_module(outputs[-1]))
        outputs = tuple(outputs)
        return outputs
