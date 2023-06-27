from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from ..component import Attention, MLPBlock, init_weights_


class AcuityStep(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.q2z_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_ffn)
        self.norm2 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, k, v):
        q = self.norm1(q + self.dropout1(self.q2z_attn(q=q, k=k, v=v)))
        q = self.norm2(q + self.dropout2(self.mlp(q)))
        return q

class AcuityLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, key_features=["res5","res4","res3"]):
        super().__init__()
        self.acuity_steps = nn.ModuleDict(dict(
            (key, AcuityStep(
                embed_dim=embed_dim, 
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn
            ))
            for key in key_features
        ))
        self.res_steps = nn.ModuleDict(dict(
            (key, AcuityStep(
                embed_dim=embed_dim, 
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn
            ))
            for key in key_features
        ))
        self.key_features = key_features

        self.fuse = nn.ModuleDict(dict(
            (key, nn.Sequential(nn.Linear(embed_dim, embed_dim//2), nn.ReLU(), nn.Linear(embed_dim//2, 1)))
            for key in key_features)
        )
        init_weights_(self.fuse)
    
    def forward(self, q, zs):
        ## Update q
        qs = []
        ws = []
        for key in self.key_features:
            k = zs[key]  ## B, hw, C
            f = self.acuity_steps[key](q=q, k=k, v=k)  ## B, nq, C
            w = self.fuse[key](f)  ## B, nq, 1
            qs.append(f)
            ws.append(w)
        ws = torch.softmax(torch.stack(ws, dim=0), dim=0)  ## n, B, nq, 1
        qs = torch.stack(qs, dim=0)  ## n, B, nq, C
        q = torch.sum(ws * qs, dim=0)  ## B, nq, C

        ## Update feats
        for key in self.key_features:
            k = zs[key]
            k = self.res_steps[key](q=k, k=q, v=q)  ## B, hw, C
            zs[key] = k

        return q, zs

class AcuityBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, key_features=["res5","res4","res3"], num_blocks=3):
        super().__init__()
        self.layers = nn.ModuleList([
            AcuityLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn,
                key_features=key_features
            )
            for _ in range(num_blocks)
        ])
        self.key_features = key_features
    
    def forward(self, q, feats):
        sizes = dict((k, v.shape) for k, v in feats.items())
        feats = dict((k, v.flatten(2).transpose(-1, -2)) for k, v in feats.items())
        for layer in self.layers:
            q, feats = layer(q, feats)
        feats = dict((k, v.transpose(-1, -2).reshape(sizes[k])) for k, v in feats.items())
        return q, feats

class FovealParallel(nn.Module):
    @configurable
    def __init__(self, num_queries=100, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2, key_features=["res5","res4","res3"]):
        super().__init__()
        self.q = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.qpe = nn.Parameter(torch.randn((1, num_queries, embed_dim)))

        self.acuity = AcuityBlock(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            dropout_attn=dropout_attn, 
            dropout_ffn=dropout_ffn, 
            key_features=key_features, 
            num_blocks=num_blocks
        )

        self.key_features = key_features
        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.bbox_head = nn.Linear(embed_dim, 4)
        self.fg_head = nn.Linear(embed_dim, 1)

        init_weights_(self.mlp)
        init_weights_(self.bbox_head)
        init_weights_(self.fg_head)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_queries":  cfg.MODEL.COMMON.NUM_QUERIES,
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks":   cfg.MODEL.MODULES.FOVEALQ.NUM_BLOCKS,
            "key_features":   cfg.MODEL.MODULES.FOVEALQ.KEY_FEATURES
        }

    def forward(self, feats, feats_pe):
        """
        No self-attn across q
        Args:
            feats: dict of B,C,Hi,Wi
            feats_pe: dict of B,C,Hi,Wi
        Returns:
            q: B, nq, C
            qpe: B, nq, C
            masks: B, nq, Hmax, Wmax
            bboxes: B, nq, 4
            fg: B, nq, 1 (logit)
        """
        low_key = self.key_features[-1]  ## higher resolution
        B, _, H, W = feats[low_key].shape
        qpe = self.qpe.expand(B, -1, -1)  ## B, nq, C

        q = self.q.expand(B, -1, -1)  ## B, nq, C
        q, feats = self.acuity(q, feats)
        low_z = feats[low_key]  ## B, C, H, W

        masks = (self.mlp(q) @ low_z.flatten(2)).unflatten(-1, (H, W))  ## B, nq, H, W
        bboxes = self.bbox_head(q)  ## B, nq, 4
        fg = self.fg_head(q)  ## B, nq, 1
        
        return q, qpe, masks, bboxes, fg
