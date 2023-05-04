import torch
from torch import Tensor, nn
from typing import List

from . import InstanceSegTransformer, mask2points, IOREncoder, AddIOR, MaskDecoder


class SRDetrDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_points = cfg.MODEL.IOR_DECODER.NUM_IOR_POINTS

        self.ins_seg_transformer = InstanceSegTransformer(cfg)
        self.ior_encoder = IOREncoder(cfg)
        self.add_ior = AddIOR(cfg)
        self.mask_decoder = MaskDecoder(cfg)

    def forward(self, feat: Tensor, ior_masks: List):
        """

        Args:
            feat: B, C, H, W
            ior_masks: list of torch.Tensor, each tensor should be N,H,W
                    the length of list is B

        Returns: logit
            masks: B, nq, 4H, 4W
            iou_scores: B, nq, 1

        """
        q, z, q_pe, z_pe = self.ins_seg_transformer(feat=feat)

        coords, labels, size = mask2points(ior_masks=ior_masks, num=self.num_points) ## B,num,2 | B,1,1 | Tuple[int,int]
        ppe = self.ins_seg_transformer.get_coord_pe(coords=coords, size=size)  ## B, num, C

        tokens = self.ior_encoder(points=ppe, z=z, z_pe=z_pe)  ## tokens: B, nt, C
        q2 = self.add_ior(token=tokens, query=q)  ## B, nq, C

        q = torch.where(labels, q2, q)  ## B, nq, C
        feat = z.transpose(-1, -2).reshape(feat.shape)  ## B, C, H, W
        return self.mask_decoder(query=q, feat=feat, q_pe=q_pe, z_pe=z_pe)
