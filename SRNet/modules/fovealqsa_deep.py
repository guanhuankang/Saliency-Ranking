import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from ..component import Attention, MLPBlock, init_weights_


class AcuityStep(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.q2z_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_attn)
        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_ffn)
        self.norm2 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, k, v):
        q = self.norm(q + self.dropout(self.q2z_attn(q=q, k=k, v=v)))
        q = self.norm2(q + self.dropout2(self.mlp(q)))
        return q

class AcuityLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.high = AcuityStep(embed_dim, num_heads, hidden_dim, dropout_attn, dropout_ffn)
        self.low  = AcuityStep(embed_dim, num_heads, hidden_dim, dropout_attn, dropout_ffn)
        self.sa    = AcuityStep(embed_dim, num_heads, hidden_dim, dropout_attn, dropout_ffn)

        self.gsa = AcuityStep(embed_dim, num_heads, hidden_dim, dropout_attn, dropout_ffn)

        self.linear = nn.Linear(2, num_heads)
        self.collect = nn.Linear(num_heads, 1)

        init_weights_(self.linear)
        init_weights_(self.collect)

    def forward(self, q, qpe, high_z, high_zpe, low_z, low_zpe):
        """

        Args:
            q: B, nq, C
            qpe: B, nq, C
            high_z: B, C, h, w
            high_zpe: B, C, h, w
            low_z: B, C, H, W
            low_zpe: B, C, H, W

        Returns:
            q: B, nq, C
        """
        B, nq, C = q.shape

        low_z = low_z.flatten(2).transpose(-1, -2)
        low_zpe = low_zpe.flatten(2).transpose(-1, -2)

        high_z = high_z.flatten(2).transpose(-1, -2)
        high_zpe = high_zpe.flatten(2).transpose(-1, -2)

        q1 = self.high(q=q+qpe, k=high_z+high_zpe, v=high_z)  ## B, nq, C
        q2 = self.low(q=q+qpe, k=low_z+low_zpe, v=low_z)  ## B, nq, C
        q = torch.stack([q1, q2], dim=-1)  ## B, nq, C, 2
        q = self.linear(q)  ## B, nq, C, num_heads
        q = q.flatten(0, 1).transpose(-1, -2)  ## B*nq, num_heads, C
        q = self.sa(q=q, k=q, v=q)  ## B*nq, nh, C
        q = self.collect(q.transpose(-1, -2)).squeeze(-1)  ## B*nq, C
        q = q.reshape(B, nq, C)

        ## Global Self-Attn
        q = self.gsa(q=q+qpe, k=q+qpe, v=q)

        return q

class AcuityBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([
            AcuityLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])

    def forward(self, q, qpe, high_z, high_zpe, low_z, low_zpe):
        for layer in self.layers:
            q = layer(q, qpe, high_z, high_zpe, low_z, low_zpe)
        return q

class Head(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024):
        super().__init__()
        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.bbox_head = nn.Linear(embed_dim, 4)
        self.fg_head = nn.Linear(embed_dim, 1)

        init_weights_(self)
    
    def forward(self, q, f):
        """
        q: B, nq, C
        f: B, C, h, w
        """
        size = f.shape[2::]
        masks = (self.mlp(q) @ f.flatten(2)).unflatten(-1, size)  ## B, nq, H, W
        bboxes = self.bbox_head(q)  ## B, nq, 4
        fg = self.fg_head(q)  ## B, nq, 1
        return {
            "masks": masks,
            "bboxes": bboxes,
            "fg": fg
        }

class FovealQSADeep(nn.Module):
    @configurable
    def __init__(self, num_queries=100, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2, key_features=["res5","res4","res3"]):
        super().__init__()
        self.q = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.qpe = nn.Parameter(torch.randn((1, num_queries, embed_dim)))

        self.layers = nn.ModuleList([
            AcuityBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn, num_blocks=num_blocks)
            for _ in range(len(key_features)-1)
        ])
        self.key_features = key_features
        self.heads = nn.ModuleList([Head(embed_dim=embed_dim, hidden_dim=hidden_dim) for _ in range(len(key_features))])

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
        keys = self.key_features
        low_z = None
        B = len(feats[keys[0]])

        q = self.q.expand(B, -1, -1)  ## B, nq, C
        qpe = self.qpe.expand(B, -1, -1)  ## B, nq, C

        q_lst = [q]
        for i, layer in enumerate(self.layers):
            high, low = keys[i], keys[i+1]
            high_z = feats[high]
            high_zpe = feats_pe[high]
            low_z = feats[low]
            low_zpe = feats_pe[low]
            q = layer(q=q, qpe=qpe, high_z=high_z, high_zpe=high_zpe, low_z=low_z, low_zpe=low_zpe)
            q_lst.append(q)
        assert len(q_lst) == len(self.heads)
        ## list of dict: masks/bboxes/fg in logits
        predictions = [head(q=q, f=low_z) for head in self.heads]
        return q, qpe, predictions
