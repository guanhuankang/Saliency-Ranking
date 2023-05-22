import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..component import Attention, MLPBlock, init_weights_

from detectron2.config import configurable


class Warp(nn.Module):
    def __init__(self, center, eps=1e-6):
        """
        forward: y=x^a
        backford: x=y^(1.0/a)
        a = -log2(norm_center)
        Args:
            center: *, 2 between [-1, 1]
        """
        super().__init__()
        self.nc = (center + 1.0) / 2.0  ## normalized
        self.nc = torch.clamp(self.nc, eps, 1.0 - eps)  ## clamp
        self.a = -torch.log2(self.nc)  ## warp_factor, (*, 2) between (0,inf)

    def forward(self, coords):
        """
        forward: y=x^a
        backford: x=y^(1.0/a)
        Args:
            coords: k, *, 2, where * should have same shape as center
                    between [-1,1]
        Returns:
            coords: k, *, 2 in [-1,1], means from which coords to be sampled
        """
        return self.warp_forward(coords)

    def warp_forward(self, coords):
        coords = (coords + 1.0) / 2.0  ## normalized [0, 1]
        coords = coords ** self.a.unsqueeze(0)  ## k, *, 2 toInCoords
        coords = (coords - 0.5) * 2.0  ## rescale to [-1, 1]
        coords = torch.clamp(coords, -1.0, 1.0)
        return coords

    def warp_backward(self, coords):
        ar = 1.0 / self.a
        coords = (coords + 1.0) / 2.0  ## normalized [0, 1]
        coords = coords ** ar.unsqueeze(0)  ## k, *, 2
        coords = (coords - 0.5) * 2.0  ## rescale to [-1, 1]
        coords = torch.clamp(coords, -1.0, 1.0)
        return coords


class FovealVision(nn.Module):
    @configurable
    def __init__(self, pad_size=7, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0):
        """

        Args:
            nq:
            pad_size:
            warp_factor: large, zoom in effect will be more obvious
        """
        super().__init__()
        self.query_to_feat_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout = nn.Dropout(p=dropout_attn)
        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.iou_head = nn.Linear(embed_dim, 1)

        self.fixation_mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.pad_size = pad_size

        init_weights_(self.query_to_feat_attn)
        init_weights_(self.norm)
        init_weights_(self.mlp)
        init_weights_(self.iou_head)

    @classmethod
    def from_config(cls, cfg):
        return {
            "pad_size": cfg.MODEL.MODULES.FOVEAL_VISION.PAD_SIZE,
            "embed_dim": cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads": cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim": cfg.MODEL.COMMON.HIDDEN_DIM,
            # "dropout_ffn": cfg.MODEL.COMMON.DROPOUT_FFN
        }

    def getGrids(self, size):
        """

        Args:
            size: tuple (H, W)

        Returns:
            grid_xy: H, W, 2
                where the first_dim means normalized x and y
                we have: left_top conner is (-1,-1) and right_bottom is (1,1)

        """
        H, W = size
        grid_x = torch.linspace(-1., 1., W).unsqueeze(0).expand(H, W)  ## H, W
        grid_y = torch.linspace(-1., 1., H).unsqueeze(1).expand(H, W)  ## H, W
        return torch.stack([grid_x, grid_y], dim=2)  ## H, W, 2

    def forward(self, q, z, q_pe, z_pe, size, pe_layer):
        """

        Args:
            q: query, B, n, C
            z: feature, B, N, C
            q_pe: query positional embedding, B, n, C
            z_pe: feature positional embedding, B, N, C
            size: tuple (H, W), where H * W = N
            pe_layer: with two function2 for positional embedding
                get_coord_pe
                get_dense_pe

        Returns:
            q_warp: warp query, B, n, n, C under n-views
            pred_masks: B, n, H, W logit, the predicted masks
            iou_scores: B, n, 1, the iou scores for each mask
            others: for debug purpose / intermedia results
        """
        assert np.prod(size) == z.shape[1], "{} != {}".format(np.prod(size), z.shape[1])
        B, n, C = q.shape
        p = self.pad_size
        global_size = (size[0] + 2 * p, size[1] + 2 * p)

        if n <= 0:
            return torch.zeros((B, n, n, C), device=q.device), torch.zeros((B, n, *size), device=q.device), torch.zeros(
                (B, n, 1), device=q.device)

        fixations = self.fixation_mlp(q + q_pe) @ (z + z_pe).transpose(-1, -2)  ## B, n, N
        fixations = torch.softmax(fixations, dim=-1).unflatten(2, sizes=size)  ## B, n, H, W
        fixations = F.pad(fixations, pad=(p, p, p, p), mode="constant", value=0.0)  ## B, n, H+2p, W+2p

        grid_xy = self.getGrids(global_size).unsqueeze(0).expand(B, -1, -1, -1).to(q.device)  ## B,H+2p, W+2p, 2
        grid_xy_reshape = grid_xy.permute(1, 2, 0, 3).flatten(0, 1)  ## HW~,B,2
        center = (fixations.unsqueeze(-1) * grid_xy.unsqueeze(1)).sum(dim=[2, 3])  ## B, n, 2
        center = torch.clamp(center, -1.0, 1.0)  ## B, n, 2

        z = z.transpose(-1, -2).unflatten(2, size)  ## B, C, H, W
        z = F.pad(z, pad=(p, p, p, p), mode="constant", value=0.0)  ## B, C, H+2p, W+2p
        z_pe = pe_layer.get_dense_pe(global_size, b=B)  ## B, C, H+2p, W+2p
        z_pe = z_pe.flatten(2).transpose(-1, -2)  ## B, HW~, C

        q_warp = []
        pred_masks = []
        iou_scores = []
        warp_masks = []
        for i in range(n):  ## For each view
            warp = Warp(center=center[:, i, :])  ## create a warp instance, (B, 2)
            warp_xy = warp(grid_xy_reshape).permute(1, 0, 2).reshape(-1, *global_size, 2)  ## B, H+2p, W+2p, 2
            reco_xy = warp.warp_backward(grid_xy_reshape).permute(1, 0, 2).reshape(-1, *global_size,
                                                                                   2)  ## B, H+2p, W+2p, 2
            warp_center = warp.warp_backward(center.transpose(0, 1)).transpose(0, 1)  ## B, n, 2
            warp_center_pe = pe_layer.get_coord_pe(warp_center, global_size)  ## B, n, C

            z_warp = F.grid_sample(z, warp_xy)  ## B, C, H+2p, W+2p
            fixation_warp = F.grid_sample(fixations, warp_xy)  ## B, n, H+2p, W+2p

            q_w = (z_warp.unsqueeze(1) * fixation_warp.unsqueeze(2)).sum(dim=[-1, -2])  ## B, n, C
            z_warp = z_warp.flatten(2).transpose(-1, -2)  ## B, N~, C
            q_w = self.norm(q_w + self.dropout(
                self.query_to_feat_attn(q=q_w + warp_center_pe, k=z_warp + z_pe, v=z_warp)))  ## B, n, C
            q_warp.append(q_w + warp_center_pe)  ## B, n, C

            q_w = self.mlp(q_w)  ## B, n, C
            m_w = (q_w @ z_warp.transpose(-1, -2)).unflatten(2, global_size)  ## B, n, H+2p, W+2p
            m = F.grid_sample(m_w, reco_xy)  ## B, n, H+2p, W+2p
            pred_masks.append(m[:, i:i + 1, p:-p, p:-p])  ## B, 1, H, W
            warp_masks.append(m_w[:, i:i+1, p:-p, p:-p])

            iou_score = self.iou_head(q_w)  ## B, n, 1
            iou_scores.append(iou_score[:, i:i + 1, :])  ## B, 1, 1
        q_warp = torch.stack(q_warp, dim=1)  ## B, n(n_views), n, C
        pred_masks = torch.cat(pred_masks, dim=1)  ## B, n, H, W
        warp_masks = torch.cat(warp_masks, dim=1)
        iou_scores = torch.cat(iou_scores, dim=1)  ## B, n, 1

        ## debug
        _, _, H, W = fixations.shape
        norm_center = (center + 1.0) / 2.0
        corr_center = (1.0+2*p/W) * norm_center - p/W
        corr_center[:, :, 1] = (1.0+2*p/H) * norm_center[:, :, 1] - p/H  ## B,n,2 [0.0,1.0]
        fixations = fixations[:, :, p:-p, p:-p]  ## B,n,H,W
        others = (fixations, corr_center)
        return q_warp, pred_masks, iou_scores, others
