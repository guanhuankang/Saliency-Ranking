import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
# from ..component import PositionEmbeddingSine
from .registry import SALIENCY_INSTANCE_SEG

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout_attn=0.0):
        super().__init__()
        self.attn_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_attn, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_attn)
        self.norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, memory, memory_mask, pos=None, query_pos=None):
        out = self.attn_layer(
            query=self.with_pos_embed(query, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask
        )[0]
        query = query + self.dropout(out)
        query = self.norm(query)

        return query

class FFNLayer(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout_ffn=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_ffn)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x):
        o = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = x + self.dropout(o)
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([
            nn.Linear(i, j)
            for i, j in zip([in_dim]+h, h+[out_dim])
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < (self.num_layers - 1) else layer(x)
        return x

@SALIENCY_INSTANCE_SEG.register()
class Mask2Former(nn.Module):
    @configurable
    def __init__(self, num_queries=100, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2, key_features=["res5","res4","res3"]):
        super().__init__()
        self.q = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.qpe = nn.Parameter(torch.randn((1, num_queries, embed_dim)))

        # self.pe_layer = PositionEmbeddingSine(embed_dim//2, normalize=True)

        self.cross_attn_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_attn=dropout_attn) for _ in
             range(num_blocks)])
        self.self_attn_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_attn=dropout_attn) for _ in
             range(num_blocks)])
        self.ffn_layers = nn.ModuleList(
            [FFNLayer(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout_ffn=dropout_ffn) for _ in
             range(num_blocks)])

        self.level_embed = nn.Embedding(len(key_features), embedding_dim=embed_dim)
        self.key_features = key_features
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)
        self.mask_embed = MLP(embed_dim, embed_dim, embed_dim, 3)
        self.bbox_head = MLP(embed_dim, embed_dim, 4, 3)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_queries":  cfg.MODEL.COMMON.NUM_QUERIES,
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks":   cfg.MODEL.MODULES.MASK2FORMER.NUM_BLOCKS,
            "key_features":   cfg.MODEL.MODULES.MASK2FORMER.KEY_FEATURES
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
            out, aux: list of dict with following fields:
                "masks": B, nq, H, W [logit]
                "scores": B, nq, 1 [logit]
                "bboxes": B, nq, 4 [logit]
                "attn_mask": BxHead, nq, HW [0,1 and detached]
        """
        mask_key = self.key_features[-1]
        mask_feature = feats[mask_key]

        B, C, H, W = mask_feature.shape
        n_keys = len(self.key_features)
        sizes = [feats[key].shape[2::] for key in self.key_features]

        q = self.q.expand(B, -1, -1)
        qpe = self.qpe.expand(B, -1, -1)

        predictions = [self.forward_prediction_head(q=q, mask_feature=mask_feature, attn_size=sizes[0])]
        for idx in range(self.num_blocks):
            key = self.key_features[int(idx % n_keys)]
            attn_mask = predictions[-1]["attn_mask"]

            # Cross-Attn: query, memory, memory_mask, pos = None, query_pos = None
            q = self.cross_attn_layers[idx](
                query=q,
                memory=feats[key].flatten(2).transpose(-1, -2) + self.level_embed.weight[idx % n_keys][None, None, :],
                memory_mask=attn_mask,
                pos=feats_pe[key].flatten(2).transpose(-1, -2),
                query_pos=qpe
            )

            # Self-Attn: query, memory, memory_mask, pos = None, query_pos = None
            q = self.self_attn_layers[idx](
                query=q,
                memory=q,
                memory_mask=None,
                pos=qpe,
                query_pos=qpe
            )

            # FFN
            q = self.ffn_layers[idx](q)

            predictions.append(self.forward_prediction_head(q=q, mask_feature=mask_feature, attn_size=sizes[(idx+1) % n_keys]))

        out = predictions[-1]
        aux = predictions[0:-1]
        return q, qpe, out, aux

    def forward_prediction_head(self, q, mask_feature, attn_size):
        """
        q: B, nq, C
        mask-feature: B, C, H, W
        attn_size: (int, int)
        """
        q = self.decoder_norm(q)
        mask_embed = self.mask_embed(q)
        scores = self.score_head(q)  ## B, nq, 1
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)  ## B, nq, H, W
        bboxes = self.bbox_head(q)  ## B, nq, 4

        attn_mask = F.interpolate(masks.detach(), size=attn_size, mode="bilinear", align_corners=False)
        attn_mask = attn_mask.sigmoid().flatten(2).unsqueeze(1).expand(-1, self.num_heads, -1, -1).flatten(0, 1).lt(0.5)
        attn_mask = attn_mask.bool().detach()  ## detach | [0,1] | BxHead,nq,hw
        return {
            "masks": masks,  ## B, nq, H, W
            "scores": scores,  ## B, nq, 1
            "bboxes": bboxes,  ## B, nq, 4
            "attn_mask": attn_mask  ## BxHead, nq, HW
        }
