import torch
import torch.nn as nn
from ..component import Attention, MLPBlock, init_weights_
from detectron2.config import configurable

class BBoxLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.q2z_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_ffn)
        self.norm3 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, z, qpe, zpe):
        q = self.norm1(q + self.dropout1(self.q2z_attn(q=q+qpe, k=z+zpe, v=z)))
        q = self.norm2(q + self.dropout2(self.self_attn(q=q+qpe, k=q+qpe, v=q)))
        q = self.norm3(q + self.dropout3(self.mlp(q)))
        return q

class BBoxDecoder(nn.Module):
    @configurable
    def __init__(self, num_queries=100, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()
        self.q = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.qpe = nn.Parameter(torch.randn((1, num_queries, embed_dim)))

        self.layers = nn.ModuleList([
            BBoxLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])

        self.bbox_head = nn.Linear(embed_dim, 4)
        self.fg_head = nn.Linear(embed_dim, 1)

        init_weights_(self.bbox_head)
        init_weights_(self.fg_head)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_queries":  cfg.MODEL.MODULES.BBOX_DECODER.NUM_QUERIES,
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks":   cfg.MODEL.MODULES.BBOX_DECODER.NUM_BLOCKS
        }

    def forward(self, z, zpe):
        """

        Args:
            z: B, hw, C
            zpe: B, hw, C

        Returns:
            q: B, nq, C
            qpe: B, nq, C
            bbox: B, nq, 4 (x, y, h, w)
            fg: B, nq, 1 (logit: fg or bg)
        """
        q = self.q
        qpe = self.qpe
        for layer in self.layers:
            q = layer(q=q, z=z, qpe=qpe, zpe=zpe)
        bbox = self.bbox_head(q)
        fg = self.fg_head(q)
        return q, qpe, bbox, fg
