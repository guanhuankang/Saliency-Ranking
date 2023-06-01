import torch.nn as nn
from detectron2.config import configurable
from ..component import MLPBlock, Attention
from ..component import init_weights_

class MaskDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.query_to_feat_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_ffn)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.feat_to_query_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout4 = nn.Dropout(p=dropout_attn)
        self.norm4 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, z, qpe, zpe):
        '''

        Args:
            q: B, nq, C
            z: B, HW, C
            qpe: B, nq, C
            zpe: B, HW, C

        Returns:
            q: B, nq, C
            z: B, HW, C

        '''
        q = self.norm1(q + self.dropout1(self.self_attn(q=q+qpe, k=q+qpe, v=q)))
        q = self.norm2(q + self.dropout2(self.query_to_feat_attn(q=q+qpe, k=z+zpe, v=z)))
        q = self.norm3(q + self.dropout3(self.mlp(q)))
        z = self.norm4(z + self.dropout4(self.feat_to_query_attn(q=z+zpe, k=q+qpe, v=q)))
        return q, z

class MaskDecoder(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MaskDecoderLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])

        self.query_to_feat_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_attn)
        self.norm = nn.LayerNorm(embed_dim)
        self.mask_mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)

        init_weights_(self.query_to_feat_attn)
        init_weights_(self.norm)
        init_weights_(self.mask_mlp)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads": cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim": cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn": cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks": cfg.MODEL.MODULES.MASK_DECODER.NUM_BLOCKS
        }

    def forward(self, q, z, qpe, zpe, size):
        """

        Args:
            q: B, nq, C
            z: B, hw, C
            qpe: B, nq, C
            zpe: B, hw, C
            size: tuple(int, int)

        Returns:
            m: B, nq, *size
        """
        for layer in self.layers:
            q, z = layer(q=q, z=z, qpe=qpe, zpe=zpe)
        q = self.norm(q + self.dropout(self.query_to_feat_attn(q=q+qpe, k=z+zpe, v=z)))
        q = self.mask_mlp(q)
        m = q @ z.transpose(-1, -2)
        m = m.unflatten(2, size)
        return m
