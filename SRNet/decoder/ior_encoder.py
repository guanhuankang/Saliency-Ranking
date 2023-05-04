import torch
import torch.nn as nn

from detectron2.config import configurable

from ..component import MLPBlock, Attention
from ..component import init_weights_


class IOREncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.points_to_feat = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
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
        q = self.norm1(q + self.dropout1(self.points_to_feat(q=q + qpe, k=z + zpe, v=z)))
        q = self.norm2(q + self.dropout2(self.self_attn(q=q + qpe, k=q + qpe, v=q)))
        q = self.norm3(q + self.dropout3(self.mlp(q)))
        return q

class IOREncoder(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, num_tokens=20, dropout_attn=0.0, dropout_ffn=0.0,
                 num_blocks=2, num_points=256):
        super().__init__()
        self.points = nn.Parameter(torch.zeros(1, num_points, embed_dim))
        self.tokens = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.tokens_pos = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

        self.layers = nn.ModuleList([
            IOREncoderLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn,
                            dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.token_emb = nn.Embedding(1, embedding_dim=embed_dim)
        self.points_emb = nn.Embedding(1, embedding_dim=embed_dim)
        self.num_tokens = num_tokens

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "num_heads": cfg.MODEL.IOR_DECODER.CROSSATTN.NUM_HEADS,
            "dropout_attn": cfg.MODEL.IOR_DECODER.CROSSATTN.DROPOUT,
            "hidden_dim": cfg.MODEL.IOR_DECODER.FFN.HIDDEN_DIM,
            "dropout_ffn": cfg.MODEL.IOR_DECODER.FFN.DROPOUT,
            "num_blocks": cfg.MODEL.IOR_DECODER.IOR_ENCODER.NUM_BLOCKS,
            "num_tokens": cfg.MODEL.IOR_DECODER.IOR_ENCODER.NUM_TOKENS,
            "num_points": cfg.MODEL.IOR_DECODER.IOR_ENCODER.NUM_IOR_POINTS
        }

    def get_token_pos(self):
        """

        Returns:
            token_pos: 1, nt, C

        """
        return self.tokens_pos

    def forward(self, points, z, z_pe):
        """

        Args:
            points: B, np, C
            z: B, HW, C
            z_pe: B, HW, C

        Returns:
            tokens: B, nt, C
        """
        B, np, C = points.shape
        nt = self.num_tokens

        q = torch.cat([self.points, self.tokens], dim=1)  ## 1, np+nt, C
        q = torch.repeat_interleave(q, B, dim=0)  ## B, np+nt, C
        q_pe = torch.cat([points, torch.repeat_interleave(self.tokens_pos, B, dim=0)], dim=1)  ## B, np+nt, C
        q_emb = torch.cat([
            torch.repeat_interleave(self.points_emb.weight, np, dim=0),
            torch.repeat_interleave(self.token_emb.weight, nt, dim=0)
        ], dim=0).unsqueeze(0)  ## 1, np+nt, C
        q_pe = q_emb + q_pe  ## B, np+nt, C

        for layer in self.layers:
            q = layer(q=q, z=z, qpe=q_pe, zpe=z_pe)

        return q[:, -nt::, :]  ## B, nt, C