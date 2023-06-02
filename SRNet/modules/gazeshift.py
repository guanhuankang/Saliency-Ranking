import torch
import torch.nn as nn
import torchvision
from detectron2.config import configurable

from ..component import Attention, MLPBlock, init_weights_


class GazeShiftLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.q2z_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_ffn)
        self.norm3 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, z, qpe, zpe, ior):
        q = self.norm1(q + self.dropout1(self.self_attn(q=q + qpe, k=q + qpe, v=q+ior)))
        q = self.norm2(q + self.dropout2(self.q2z_attn(q=q + qpe, k=z + zpe, v=z)))
        q = self.norm3(q + self.dropout3(self.mlp(q)))
        return q

class GazeShift(nn.Module):
    @configurable
    def __init__(self, sigma=10.0, kernel_size=5, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GazeShiftLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.peripheral_vision = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.ior_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.saliency_head = nn.Linear(embed_dim, 1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,

            "num_blocks":   cfg.MODEL.MODULES.GAZE_SHIFT.NUM_BLOCKS,
            "sigma":        cfg.MODEL.MODULES.GAZE_SHIFT.SIGMA,
            "kernel_size":  cfg.MODEL.MODULES.GAZE_SHIFT.KERNEL_SIZE
        }

    def getGazeMap(self, bbox, size):
        """
        f(x) = exp(-||x-x0||^2_{\sigma}), where sigma=diag(1/w^2, 1/h^2)
        Args:
            bbox: B, 1, 4
            size: Tuple(int, int)

        Returns:
            map: B, 1, h, w
        """
        H, W = size
        x = torch.linspace(0.0, 1.0, W, device=bbox.device).unsqueeze(0).expand(H, W)
        y = torch.linspace(0.0, 1.0, H, device=bbox.device).unsqueeze(1).expand(H, W)
        x = x.unsqueeze(0).expand(len(bbox), H, W)
        y = y.unsqueeze(0).expand(len(bbox), H, W)
        xy = torch.stack([x, y], dim=1)  ## B, 2, H, W
        c = bbox[:, 0, 0:2].unsqueeze(2).unsqueeze(3)  ## B, 2, 1, 1
        rep_w = 1.0 / (bbox[:, :, -1].unsqueeze(-1).unsqueeze(-1) + 1e-6)  ## B, 1, 1, 1
        rep_h = 1.0 / (bbox[:, :, -2].unsqueeze(-1).unsqueeze(-1) + 1e-6)  ## B, 1, 1, 1
        e = (xy - c) * torch.cat([rep_w, rep_h], dim=1)  ## B, 2, H, W
        e = (e**2).sum(dim=1, keepdims=True)  ## B, 1, H, W
        e = torch.exp(-e)  ## B, 1, H, W
        return e

    def forward(self, q, z, qpe, zpe, q_vis, bbox, size):
        """

        Args:
            q: B, nq, C
            z: B, hw, C
            qpe: B, nq, C
            zpe: B, hw, C
            q_vis: B, nq, 1 (int: 0-n)
            bbox: B, nq, 4 [xyhw]
            size: Tuple(h, w)

        Returns:
            saliency: B, nq, 1 (logit)
        """
        device = q.device
        gaze_map = self.getGazeMap(torch.stack([
            torch.sigmoid(bb[vis.argmax(), :]) if vis.max() > .5 else torch.ones(4, device=device)*0.5
            for bb, vis in zip(bbox, q_vis[:, :, 0])
        ], dim=0).unsqueeze(1), size=size)  ## B, 1, h, w
        z = z.transpose(-1, -2).unflatten(2, size)  ## B, C, h, w
        z = gaze_map * z + (1.0 - gaze_map) * self.peripheral_vision(z)
        z = z.flatten(2).transpose(-1, -2)  ## B, hw, C

        ior = (q_vis / (q_vis.max(dim=1)[0].unsqueeze(1) + 1e-6)) * self.ior_embedding  ## B, nq, C
        for layer in self.layers:
            q = layer(q=q, z=z, qpe=qpe, zpe=zpe, ior=ior)
        return self.saliency_head(q)  ## B, nq, 1
