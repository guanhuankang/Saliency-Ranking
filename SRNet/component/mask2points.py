import torch
import torch.nn as nn
import math
from typing import Any, Tuple, List


def pointSelection(mask: torch.Tensor, num: int) -> torch.Tensor:
    """
    !!! Maybe SLOW !!! Due to * dimension is not in parallel
    sample points from mask where mask_{i,j} > 0.5 or True
    mask should not be empty
    return indices between 0 and (H-1)/(W-1)
    Args:
        mask: *, H, W
        num: int, number of points to be sampled

    Returns:
        indices: *, num, 2 (randomly, dtype=torch.long)

    """
    if mask.dtype != torch.bool:
        mask = mask > .5
    pre_size, size = mask.shape[0:-2], mask.shape[-2::]
    ret_size = pre_size + (num, 2)  ## *, num, 2
    device = mask.device

    indices = torch.empty(ret_size, dtype=torch.long, device=device)  ## *, num, 2
    indices = indices.reshape(-1, num, 2)
    mask = mask.reshape(-1, *size)  ## -1, H, W
    for i in range(len(mask)):  ## Maybe SLOW, fill in indices mask by mask
        idx_h, idx_w = torch.where(mask[i])  ## torch.Tensor 1-dim
        idx = torch.randperm(len(idx_h), device=device).repeat(math.ceil(num / len(idx_h)))[0:num]
        indices[i, :, 0] = idx_h[idx]
        indices[i, :, 1] = idx_w[idx]
    return indices.reshape(ret_size)


def mask2points(ior_masks: List, num: int):
    """

    Args:
        num:
        ior_masks: list of torch.Tensor, each tensor should be N,H,W
                    the length of list is B

    Returns:
        indices: B, num, 2 (torch.Tensor)
        labels: B, 1, 1 (torch.bool) indicating it is valid points indices or not
        size: H, W

    """
    indices = []
    labels = []
    H, W = 0, 0
    for m_ in ior_masks:
        m = m_.gt(0.5).float()
        H, W = m.shape[-2::]
        if m.sum() < 0.5:
            labels.append(torch.zeros((1, 1, 1), device=m.device))
            indices.append(torch.zeros((1, num, 2), device=m.device, dtype=torch.long))
        else:
            m = m.sum(dim=0, keepdims=True) ## 1, H, W
            labels.append(torch.ones((1, 1, 1), device=m.device))
            indices.append(pointSelection(mask=m, num=num)) ## 1, num, 2
    indices = torch.cat(indices, dim=0)  ## B, num, 2
    labels = torch.cat(labels, dim=0).gt(.5)  ## B, 1, 1, torch.bool
    return indices, labels, (H, W)
