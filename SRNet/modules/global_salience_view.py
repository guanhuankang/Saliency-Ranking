import torch.nn as nn
from detectron2.config import configurable


from ..component import MLPBlock, Attention
from ..component import init_weights_

class GSVLayer(nn.Module):
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


class GlobalSalienceView(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GSVLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn, dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.obj_head = nn.Linear(embed_dim, 1)
        self.gso_head = nn.Sequential(
            MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )

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

    def forward(self, q, z, q_pe, z_pe):
        """

        Args:
            q: query, B, n, C
            z: feature, B, N, C
            q_pe: query positional embedding, B, n, C
            z_pe: feature positional embedding, B, N, C

        Returns:
            q: B, n, C
            z: B, N, C
            obj_scores: B, n, 1 logit
            sal_scores: B, n, 1 logit the most salient scores
        """
        for layer in self.layers:
            q, z = layer(q=q, z=z, q_pe=q_pe, z_pe=z_pe)
        obj_scores = self.obj_head(q)  ## B, n, 1 logit  -- sigmoid
        sal_scores = self.gso_head(obj_scores.sigmoid() * q)  ## B, n, 1 logit -- softmax
        return q, z, obj_scores, sal_scores