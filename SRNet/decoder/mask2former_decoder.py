import torch
import torch.nn as nn
from typing import Tuple, Any, List

from detectron2.config import configurable
from torch import Tensor

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

class Mask2FromerDecoderLayer(nn.Module):
    def __init__(self, num_inputs=3, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            InstanceSegBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_inputs)
        ])

    def forward(self, q, qpe, z, zpe):
        """

        Args:
            q: B, nq, C
            qpe: B, nq, C
            z: list of feat: B, hw_{i}, C
            zpe: list of feat: B, hw_{i}, C

        Returns:
            q: B, nq, C
        """
        assert len(z) == len(zpe), "len of z-{} should be equal zpe-{}".format(len(z), len(zpe))
        assert len(z) == len(self.layers), "len_z-{} != len_layers-{}".format(len(z), len(self.layers))

        for i, layer in enumerate(self.layers):
            q = layer(q=q, z=z[i], qpe=qpe, zpe=zpe[i])
        return q

class Mask2FormerDecoder(nn.Module):
    """
    Modified Mask2FromerDecoder:
        1. without mask attention
        2. use 1/32, 1/16, 1/8 as k,v
    """
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_queries=20,
                 num_blocks=3, feature_keys=["res5","res4","res3"]):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        # self.key_scale = nn.Parameter(torch.ones(embed_dim))

        self.layers = nn.ModuleList([
            Mask2FromerDecoderLayer(
                num_inputs=len(feature_keys),
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)
        self.num_queries = num_queries
        self.feature_keys = feature_keys

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.DECODER.EMBED_DIM,
            "num_heads": cfg.MODEL.DECODER.NUM_HEADS,
            "hidden_dim": cfg.MODEL.DECODER.HIDDEN_DIM,
            "dropout_attn": cfg.MODEL.DECODER.DROPOUT_ATTN,
            "dropout_ffn": cfg.MODEL.DECODER.DROPOUT_FFN,
            "num_queries": cfg.MODEL.DECODER.NUM_QUERIES,
            "num_blocks": cfg.MODEL.DECODER.NUM_BLOCKS,
            "feature_keys": cfg.MODEL.DECODER.FEATURE_KEYS
        }

    def get_dense_pe(self, size: Tuple[int, int], b=1) -> torch.Tensor:
        '''

        Args:
            size: Tuple[int, int]
            b: batch_size

        Returns:
             dense_pe: b x C x H x W

        '''
        return torch.repeat_interleave(
            self.pe_layer(size).unsqueeze(0),
            b,
            dim=0
        )  # * self.key_scale.view(1, -1, 1, 1)

    def get_coord_pe(self, coords: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        '''

        Args:
            coords: B, *, 2

        Returns:
            coords_pe: B, *, C

        '''
        return self.pe_layer.forward_with_coords(coords, size)  # * self.key_scale.view(-1)

    def get_query_pos(self, b=1):
        return torch.repeat_interleave(self.query_pos, b, dim=0)

    def forward(self, feats: dict, deep_supervision=False):
        """

        Args:
            feats: dict with keys as follow (from neck net)
                res5: B,C,h,w
                res4: B,C,2h,2w
                res3: B,C,4h,4w
                ...
            deep_supervision: bool

        Returns:
            q: B, nq, C
            z: B, HW, C
            q_pe: B, nq, C
            z_pe: B, HW, C
        """

        z = [feats[k] for k in self.feature_keys]

        _, nq, C = self.query.shape
        B = len(z[0])

        q = torch.repeat_interleave(self.query, B, dim=0)  ## B, nq, C
        qpe = self.get_query_pos(B)  ## B, nq, C
        zpe = [self.get_dense_pe(x.shape[2::], B).flatten(2).transpose(-1, -2) for x in z]
        z = [x.flatten(2).transpose(-1, -2) for x in z]

        qs = [q]  ## for deep supervision
        for layer in self.layers:
            q = layer(q=q, qpe=qpe, z=z, zpe=zpe)
            qs.append(q)

        if deep_supervision:
            return q, qs
        else:
            return q
