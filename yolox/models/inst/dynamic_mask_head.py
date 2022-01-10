import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from yolox.utils import aligned_bilinear
from ..network_blocks import get_activation
from ..losses import DiceLoss


class DynamicMaskHead(nn.Module):
    def __init__(self,
                 num_layers=3,
                 in_channel=8,
                 out_channel=8, 
                 mask_stride_out=4,
                 disable_rel_coords=False,
                 soi=[64, 128, 256],
                 act='silu',
                 loss_dict=dict(loss_mask_weight=5.0)):

        super(DynamicMaskHead, self).__init__()
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mask_stride_out = mask_stride_out
        self.disable_rel_coords = disable_rel_coords
        self.register_buffer("size_of_interest", torch.tensor(soi))
        self.act = get_activation(act)
        self.weight_nums, self.bias_nums, self.num_gen_params\
            = self.generate_dynamic_filters()
        self.loss_dict = loss_dict
        
        self.diceloss = DiceLoss(reduction="sum", loss_type="sqrt", 
                loss_weight=loss_dict["loss_mask_weight"])
        
    def generate_dynamic_filters(self):
        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channel + 2) * self.out_channel) 
                else:
                    weight_nums.append(self.in_channel * self.out_channel)
                bias_nums.append(self.in_channel)
            elif l == self.num_layers - 1:
                weight_nums.append(self.in_channel * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.out_channel * self.in_channel)
                bias_nums.append(self.in_channel)
        num_gen_params = sum(weight_nums) + sum(bias_nums)
        return weight_nums, bias_nums, num_gen_params

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.ndim == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts)

        return weight_splits, bias_splits 

    def mask_heads_forward(self, features, weights, biases, num_insts):
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, 
                         stride=1, padding=0,
                         groups=num_insts)
            if i < n_layers - 1:
                x = self.act(x)
        return x

    def mask_heads_forward_with_coords(
        self, mask_feats, mask_feat_stride, extra_infos
    ):
        hsize, wsize = mask_feats.shape[2:]        
        yv, xv = torch.meshgrid((torch.arange(hsize, device=mask_feats.device),
                         torch.arange(wsize, device=mask_feats.device)))
        yv, xv = yv.reshape(-1), xv.reshape(-1)
        grid = torch.stack((yv, xv), dim=1) * mask_feat_stride + 0.5 * mask_feat_stride
        grid = grid.type(mask_feats.dtype)

        num_pos_masks = extra_infos["num_fg"]
        img_inds = extra_infos["img_inds"]
        mask_head_params = extra_infos["pos_ctl_preds"]

        hsize, wsize = mask_feats.shape[2:]
        if not self.disable_rel_coords:
            pos_locations = extra_infos["pos_grid_positions"]
            relative_coords = pos_locations.view(-1, 1, 2) - grid.view(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1)
            soi = (self.size_of_interest.type(mask_feats.dtype))[extra_infos["pos_fpn_levels"]]
            relative_coords = relative_coords / soi.view(-1, 1, 1)
            relative_coords = relative_coords.type(mask_feats.dtype)
            mask_head_inputs = torch.cat([

                relative_coords, mask_feats[img_inds].reshape(num_pos_masks, self.in_channel, hsize * wsize)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[img_inds].reshape(num_pos_masks, self.in_channel, hsize * wsize)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, hsize, wsize)

        weights, biases = self.parse_dynamic_params(
            mask_head_params, self.out_channel, 
            self.weight_nums, self.bias_nums
        )


        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, num_pos_masks)
        mask_logits = mask_logits.reshape(-1, 1, hsize, wsize)
        mask_logits = aligned_bilinear(mask_logits, mask_feat_stride // self.mask_stride_out)

        return  mask_logits

    def get_gt_masks(self, gt_masks, gt_inds, resize_shape, device, dtype):
        gt_masks = np.concatenate(gt_masks, 0)
        gt_masks = torch.from_numpy(gt_masks).to(device).type(dtype)
        gt_masks = gt_masks[gt_inds]
        gt_masks = gt_masks[:, None]
        # resized_gt_masks = aligned_bilinear(gt_masks, stride_out)
        resized_gt_masks = nn.functional.interpolate(
            gt_masks, size=resize_shape, mode="bilinear", align_corners=False
        )
        return resized_gt_masks.squeeze(1)

        
    def forward(self, mask_feats, mask_feat_stride, extra_infos):
        if self.training:
            gt_inds = extra_infos["gt_inds"]
            resize_shape = (int(mask_feats.shape[2] / (self.mask_stride_out / mask_feat_stride)),
                int(mask_feats.shape[3] / (self.mask_stride_out / mask_feat_stride)))
            gt_masks = self.get_gt_masks(extra_infos["gt_masks"], gt_inds, 
                                         resize_shape,
                                         mask_feats.device, mask_feats.dtype)

            if len(gt_masks) == 0:
                return dict(
                    loss_mask=mask_feats.sum() * 0.
                )
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, extra_infos
                )
                mask_scores = mask_logits.sigmoid()
                loss_mask = self.diceloss(mask_scores, gt_masks, extra_infos["num_fg"])
                loss_infos = dict(loss_mask=loss_mask)
                return loss_infos
        else:
            mask_logits = self.mask_heads_forward_with_coords(
                mask_feats, mask_feat_stride, extra_infos
            ).sigmoid()
            mask_logits = nn.functional.interpolate(
                mask_logits, scale_factor=self.mask_stride_out,
                mode="bilinear", align_corners=True
            )
            mask_logits = mask_logits.squeeze(1)
            img_inds = extra_infos["img_inds"]
            bbox_preds = extra_infos["pos_bbox_preds"]
            mask_logits_ = []
            bbox_preds_ = []

            for img_id in img_inds.unique().tolist():
                id_mask = img_inds == img_id
                mask_logits_.append(mask_logits[id_mask])
                bbox_preds_.append(bbox_preds[id_mask])

            return bbox_preds_, mask_logits_
            