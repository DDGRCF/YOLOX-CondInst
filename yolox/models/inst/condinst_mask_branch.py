import torch.nn as nn
from functools import partial
from ..network_blocks import BaseConv, DWConv
from yolox.utils import aligned_bilinear

class CondInstMaskBranch(nn.Module):
    def __init__(
        self,
        in_channels=[256, 512, 1024],
        width=1.0,
        feat_channel=128,
        out_channel=8,
        num_convs=4,
        depthwise=False,
        out_stride=8
    ):
        super().__init__()
        self.width = width
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_convs = num_convs
        self.out_stride = out_stride
        Conv = DWConv if depthwise else BaseConv
        self.refine_modules = nn.ModuleList()
        norm_func = partial(nn.init.kaiming_uniform_, a=1)
        for c in in_channels:
            self.refine_modules.append(
                Conv(int(c * width), int(feat_channel * width), 3, 1, init_func=norm_func)
            )
        mask_modules = nn.ModuleList() 
        for _ in range(num_convs):
            mask_modules.append(
                Conv(int(feat_channel * width), int(feat_channel * width), 3, 1, init_func=norm_func)
            )
        mask_modules.append(
            Conv(int(feat_channel * width), out_channel, 3, 1, init_func=norm_func)
        )
        self.add_module('mask_modules', nn.Sequential(*mask_modules))

    def forward(self, features):
        for i, f in enumerate(features):
            if i == 0:
                x = self.refine_modules[i](f)
            else:
                x_p = self.refine_modules[i](f) 

                target_h, target_w = x.shape[2:]
                h, w = x_p.shape[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        return self.mask_modules(x)

            
        

