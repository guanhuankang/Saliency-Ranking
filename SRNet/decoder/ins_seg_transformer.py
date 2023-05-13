import torch
import torch.nn as nn
from typing import Tuple

from detectron2.config import configurable

from ..component import MLPBlock, Attention
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

        init_weights_(self)

    def forward(self, q, z, qpe, zpe):
        q = self.norm1(q + self.dropout1(self.q2z(q=q + qpe, k=z + zpe, v=z)))
        q = self.norm2(q + self.dropout2(self.self_attn(q=q + qpe, k=q + qpe, v=q)))
        q = self.norm3(q + self.dropout3(self.mlp(q)))
        return q


class InstanceSegTransformer(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_queries=20,
                 num_blocks=4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        # self.key_scale = nn.Parameter(torch.ones(embed_dim))

        self.layers = nn.ModuleList([
            InstanceSegBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn,
                             dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)
        self.cls_emb = nn.Embedding(1, embedding_dim=embed_dim)  ## class emb
        self.obj_emb = nn.Embedding(1, embedding_dim=embed_dim)  ## object emb
        self.num_queries = num_queries

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "num_heads": cfg.MODEL.IOR_DECODER.CROSSATTN.NUM_HEADS,
            "hidden_dim": cfg.MODEL.IOR_DECODER.FFN.HIDDEN_DIM,
            "dropout_attn": cfg.MODEL.IOR_DECODER.CROSSATTN.DROPOUT,
            "dropout_ffn": cfg.MODEL.IOR_DECODER.FFN.DROPOUT,
            "num_queries": cfg.MODEL.IOR_DECODER.INSTANCE_SEG.NUM_QUERIES,
            "num_blocks": cfg.MODEL.IOR_DECODER.INSTANCE_SEG.NUM_BLOCKS,
            # "spatial_pe_size": (cfg.MODEL.IOR_DECODER.LEARNABLE_PE.HEIGHT, cfg.MODEL.IOR_DECODER.LEARNABLE_PE.WIDTH)
        }

    def get_dense_pe(self, size: Tuple[int, int]) -> torch.Tensor:
        '''

        Args:
            size: Tuple[int, int]

        Returns:
             dense_pe: 1 x C x H x W

        '''
        return self.pe_layer(size).unsqueeze(0)  # * self.key_scale.view(1, -1, 1, 1)

    def get_coord_pe(self, coords: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        '''

        Args:
            coords: B, *, 2

        Returns:
            coords_pe: B, *, C

        '''
        return self.pe_layer.forward_with_coords(coords, size)  # * self.key_scale.view(-1)

    def get_query_emb(self, plus_pos=True):
        """

        Returns:
            query_emb: 1, nq, C

        """
        obj_emb = torch.repeat_interleave(self.obj_emb.weight, self.num_queries - 1, dim=0)  ## nq-1, C
        cls_emb = self.cls_emb.weight  ## 1, C
        q_emb = torch.cat([obj_emb, cls_emb], dim=0).unsqueeze(0)  ## 1, nq, C
        if plus_pos:
            q_emb = q_emb + self.query_pos
        return q_emb

    def get_query_pos(self):
        return self.query_pos

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            feat: B, C, H, W

        Returns:
            q: B, nq, C
            z: B, HW, C
            q_pe: B, nq, C
            z_pe: B, HW, C
        """
        _, nq, C = self.query.shape
        B, C, H, W = feat.shape
        size = (H, W)

        ## prepare q, z and qpe, zpe (positional embedding)
        q = torch.repeat_interleave(self.query, B, dim=0)  ## B, nq, C
        z = feat.flatten(2).transpose(-1, -2)  ## B, HW, C
        qpe = torch.repeat_interleave(self.get_query_pos(), B, dim=0)  ## B, nq, C
        zpe = torch.repeat_interleave(self.get_dense_pe(size), B, dim=0).flatten(2).transpose(-1, -2)  ## B, HW, C

        ## feed to layers
        for layer in self.layers:
            q = layer(q=q, z=z, qpe=qpe, zpe=zpe)

        return q, z, qpe, zpe
