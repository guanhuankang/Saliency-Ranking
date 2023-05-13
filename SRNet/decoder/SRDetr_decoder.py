import torch
from torch import Tensor, nn
from typing import List

from . import InstanceSegTransformer, mask2points, IOREncoder, AddIOR, MaskDecoder


class SRDetrDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_points = cfg.MODEL.IOR_DECODER.IOR_ENCODER.NUM_IOR_POINTS

        self.ins_seg_transformer = InstanceSegTransformer(cfg)
        self.mask_decoder = MaskDecoder(cfg)

    def forward(self, feat: Tensor):
        """
        NOTE: Instance_Seg branch 

        Args:
            feat: B, C, H, W

        Returns: logit
            masks: B, nq, 4H, 4W
            iou_scores: B, nq, 1
            obj_scores: B, nq, 1
        """
        q, z, q_pe, z_pe = self.ins_seg_transformer(feat=feat)
        masks, iou_socres, obj_scores = self.mask_decoder(query=q, feat=feat, q_pe=q_pe, z_pe=z_pe)
        return masks, iou_socres, obj_scores

