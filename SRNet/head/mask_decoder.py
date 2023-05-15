import torch
import torch.nn as nn
from typing import Tuple

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

    def forward(self, q, z, q_pe, z_pe):
        '''

        Args:
            q: B, nq, C
            z: B, HW, C
            q_pe: B, nq, C
            z_pe: B, HW, C

        Returns:
            q: B, nq, C
            z: B, HW, C

        '''
        q = self.norm1(q + self.dropout1(self.self_attn(q=q+q_pe, k=q+q_pe, v=q)))
        q = self.norm2(q + self.dropout2(self.query_to_feat_attn(q=q+q_pe, k=z+z_pe, v=z)))
        q = self.norm3(q + self.dropout3(self.mlp(q)))
        z = self.norm4(z + self.dropout4(self.feat_to_query_attn(q=z+z_pe, k=q+q_pe, v=q)))
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
        self.iou_head = nn.Linear(embed_dim, 1)
        self.obj_head = nn.Linear(embed_dim, 1)

        init_weights_(self.query_to_feat_attn)
        init_weights_(self.norm)
        init_weights_(self.mask_mlp)
        init_weights_(self.iou_head)
        init_weights_(self.obj_head)

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

    def forward(self, query, feat, q_pe, z_pe):
        '''

        Args:
            query: B, nq, C
            feat: B, C, H, W
            q_pe: B, nq, C
            z_pe: B, HW, C

        Returns: logit
            masks: B, nq, H, W logit
            iou_scores: B, nq, 1 logit
            obj_scores: B, nq, 1 logit
        '''
        B, C, H, W = feat.shape
        z = feat.flatten(2).transpose(-1, -2)  ## B, HW, C
        for layer in self.layers:
            query, z = layer(query, z, q_pe, z_pe)
        query = self.norm(query+self.dropout(self.query_to_feat_attn(q=query+q_pe, k=z+z_pe, v=z)))  ## B, nq, C

        masks = (self.mask_mlp(query) @ z.transpose(-1, -2)).reshape(B, -1, H, W)  ## B, C, H, W
        iou_socres = self.iou_head(query)  ## B, nq, 1
        obj_scores = self.obj_head(query)  ## B, nq, 1

        return masks, iou_socres, obj_scores
