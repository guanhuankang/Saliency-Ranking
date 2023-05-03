import torch
import torch.nn as nn
from typing import Any, Tuple

from detectron2.config import configurable

from .sam import MLPBlock, Attention
from ..component import PositionEmbeddingRandom, init_weights_


class InstanceSegBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.q2z = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_ffn)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.z2q = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout4 = nn.Dropout(p=dropout_attn)
        self.norm4 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, z, qpe, zpe):
        q = self.norm1(q + self.dropout1(self.q2z(q=q+qpe, k=z+zpe, v=z)))
        q = self.norm2(q + self.dropout2(self.self_attn(q=q+qpe, k=q+qpe, v=q)))
        q = self.norm3(q + self.dropout3(self.mlp(q)))
        z = self.norm4(z + self.dropout4(self.z2q(q=z+zpe, k=q+qpe, v=q)))
        return q, z
class InstanceSegTransformer(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_queries=20, num_blocks=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.layers = nn.ModuleList([
            InstanceSegBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim//2)
        self.cls_emb = nn.Embedding(1, embedding_dim=embed_dim) ## class emb
        self.obj_emb = nn.Embedding(1, embedding_dim=embed_dim) ## object emb

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "num_heads": cfg.MODEL.IOR_DECODER.CROSSATTN.NUM_HEADS,
            "hidden_dim": cfg.MODEL.IOR_DECODER.FFN.HIDDEN_DIM,
            "dropout_attn": cfg.MODEL.IOR_DECODER.CROSSATTN.DROPOUT,
            "dropout_ffn": cfg.MODEL.IOR_DECODER.FFN.DROPOUT,
            "num_queries": cfg.MODEL.IOR_DECODER.NUM_QUERIES,
            "num_blocks": cfg.MODEL.IOR_DECODER.NUM_BLOCKS
        }

    def get_dense_pe(self, size: Tuple[int, int]) -> torch.Tensor:
        '''

        Args:
            size: Tuple[int, int]

        Returns:
             dense_pe: 1 x C x H x W

        '''
        return self.pe_layer(size).unsqueeze(0)

    def get_coord_pe(self, coords: torch.Tensor, size:Tuple[int, int]) -> torch.Tensor:
        '''

        Args:
            coords: B, *, 2

        Returns:
            coords_pe: B, *, C

        '''
        return self.pe_layer.forward_with_coords(coords, size)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        Args:
            feat: B, C, H, W

        Returns:
            query: B, num_queries, C
            feat: updated feat - B, C, H, W

        '''
        _, L, C = self.query.shape
        B, C, H, W = feat.shape
        size = (H, W)

        ## prepare q, z and qpe, zpe (positional embedding)
        q = torch.repeat_interleave(self.query, B, dim=0) ## B, nq, C
        z = feat.flatten(2).transpose(-1, -2) ## B, HW, C
        qpe = torch.ones_like(q) * self.obj_emb.weight ## B, nq, C
        qpe[:, -1, :] = self.cls_emb.weight ## set the last token as class token
        zpe = torch.repeat_interleave(self.get_dense_pe(size), B, dim=0) ## B, C, H, W

        ## feed to layers
        for layer in self.layers:
            q, z = layer(q, z, qpe, zpe)

        feat = z.transpose(-1, -2).reshape(B, C, H, W)
        return feat, q
