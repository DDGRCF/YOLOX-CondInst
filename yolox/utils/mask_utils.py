import torch.nn.functional as F
import cv2
from typing import Tuple

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return 1
    h, w = tensor.shape[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode='replicate')
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True # TODO: test True or False
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )
    return tensor[..., :oh - 1, :ow - 1]

def resize_mask(input, dsize: Tuple[int], interpolation=cv2.INTER_LINEAR):
    masks = cv2.resize(input, dsize, interpolation=interpolation)
    if masks.ndim == 2:
        masks = masks[..., None]
    return masks