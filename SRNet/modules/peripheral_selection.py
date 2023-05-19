import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..component import Attention, MLPBlock, init_weights_

from detectron2.config import configurable

class SA(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_ffn)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, x_pe=None):
        if isinstance(x_pe, type(None)):
            x = self.norm1(x + self.dropout1(self.self_attn(q=x, k=x, v=x)))
        else:
            x = self.norm1(x + self.dropout1(self.self_attn(q=x+x_pe, k=x+x_pe, v=x)))
        return self.norm2(x + self.dropout2(self.mlp(x)))

class PeripheralSelection(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([
            SA(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.inh = nn.Parameter(torch.randn(1, embed_dim))
        self.sel_head = nn.Linear(embed_dim, 1)
        self.inh_head = nn.Linear(embed_dim, 1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.HEAD.EMBED_DIM,
            "num_heads": cfg.MODEL.HEAD.NUM_HEADS,
            "dropout_attn": cfg.MODEL.HEAD.DROPOUT_ATTN,
            "hidden_dim": cfg.MODEL.HEAD.HIDDEN_DIM,
            "dropout_ffn": cfg.MODEL.HEAD.DROPOUT_FFN,
            "num_blocks": cfg.MODEL.HEAD.NUM_BLOCKS
        }

    def forward(self, q_w, q_m):
        """

        Args:
            q_w: query_warp B, m, n, C (m=num_views, n=num_queries)
            q_m: query_mask/inhibition B, m, n (01 mask, 1 means inhibition, 0 means no inh)

        Returns:
            inh_scores: B, m, n (logit -- sigmoid dim=-1, lt0.5 means is inhibition)
            sel_scores: B, m, n (logit -- softmax dim=-1, selection)
        """
        B, m, n = q_m.shape
        q_m = q_m.unsqueeze(-1) * self.inh.reshape(1, 1, 1, -1)  ## B, m, n, C
        q_w = q_w + q_m  ## B, m, n, C
        q_w = q_w.flatten(0, 1)  ## B * m, n, C
        for layer in self.layers:
            q_w = layer(q_w)
        sel = self.sel_head(q_w).unflatten(0, (B, m))  ## B, m, n, 1
        inh = self.inh_head(q_w).unflatten(0, (B, m))  ## B, m, n, 1
        return inh.squeeze(-1), sel.squeeze(-1)
