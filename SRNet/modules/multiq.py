import numpy as np
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

class Head(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)
        self.mask_embed = MLP(embed_dim, embed_dim, embed_dim, 3)
        self.bbox_head = MLP(embed_dim, embed_dim, 4, 3)
        self.num_heads = num_heads
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, q, mask_feature, attn_size):
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

class SeeThroughMultiQ(nn.Module):
    def __init__(self, embed_dim=256, mlp_layers=2):
        super().__init__()
        self.conv_z = nn.Conv2d(embed_dim, embed_dim, 1)
        self.fc_q = nn.Linear(embed_dim, embed_dim)

        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.g1 = nn.Sequential(MLP(embed_dim, embed_dim, embed_dim, mlp_layers), nn.LayerNorm
                                (embed_dim), nn.Sigmoid())
        self.g2 = nn.Sequential(MLP(embed_dim, embed_dim, embed_dim, mlp_layers), nn.LayerNorm
                                (embed_dim), nn.Sigmoid())
        self.fc1 = MLP(embed_dim, embed_dim, embed_dim, mlp_layers)
        self.fc2 = MLP(embed_dim, embed_dim, embed_dim, mlp_layers)
        self.fc_out = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, q, qpe, feat, feat_pe, bboxes, grid_embedding):
        """
        Args:
            q, qpe: B, nq, C
            feat, feat_pe: B, C, H, W
            bboxes: B, nq, 4 [xyhw in [0,1]]
            grid_embedding: s1, s2, C
        Return:
            q_split: B, nq, s*s, C
        """
        s1, s2, _ = grid_embedding.shape
        n_grids = int(s1 * s2)

        q_split = self.fc_q(q + qpe).unsqueeze(2).expand(-1, -1, n_grids, -1)  ## B, nq, s*s, C
        q_split = q_split + grid_embedding.flatten(0, 1).unsqueeze(0).unsqueeze(1) ## B, nq, s*s, C
        roi_feat = self.ROI_sample(x=self.conv_z(feat+feat_pe), bboxes=bboxes, size=(s1, s2))  ## B, nq, C, s1, s2
        roi_feat = roi_feat.flatten(3).transpose(-1, -2)  ## B, nq, s*s, C
        k = self.linear1(q_split) * self.linear2(roi_feat)  ## B, nq, s*s, C

        q = q.unsqueeze(2).expand(-1, -1, n_grids, -1) + grid_embedding.flatten(0, 1).unsqueeze(0).unsqueeze(1)  ## B, nq, s*s, C
        v = self.ROI_sample(x=feat, bboxes=bboxes, size=(s1, s2)).flatten(3).transpose(-1, -2)  ## B, nq, s*s, C
        out = self.g1(k) * self.fc1(q) + self.g2(k) * self.fc2(v)
        out = self.fc_out(out)  ## B, nq, s*s, C
        return out

    def ROI_sample(self, x, bboxes, size):
        """
        Args:
            x: B, C, H, W
            bboxes: B, nq, 4 [xyhw in [0,1]]
            size: (s1, s2)
        Return:
            roi: B, nq, C, s1, s2
        """
        # bboxes = 2.0 * bboxes - 1.0  ## [-1.0, 1.0]
        s1, s2 = size
        B, nq, _ = bboxes.shape

        hs = torch.arange(s1, device=x.device, dtype=x.dtype)  ## s1
        ws = torch.arange(s2, device=x.device, dtype=x.dtype)  ## s2
        h_slice = (bboxes[:, :, 2] / s1).unsqueeze(2)  ## B, nq, 1
        w_slice = (bboxes[:, :, 3] / s2).unsqueeze(2)  ## B, nq, 1
        hs = h_slice * hs[None, None, :] + h_slice / 2.0  ## B, nq, s1
        ws = w_slice * ws[None, None, :] + w_slice / 2.0  ## B, nq, s2
        ys = hs.unsqueeze(3).expand(-1, -1, s1, s2)  ## B, nq, s1, s2
        xs = ws.unsqueeze(2).expand(-1, -1, s1, s2)  ## B, nq, s1, s2

        xs = xs + bboxes[:, :, 0, None, None] - bboxes[:, :, 3, None, None]/2.0
        ys = ys + bboxes[:, :, 1, None, None] - bboxes[:, :, 2, None, None]/2.0
        xs = torch.clamp(2.0 * xs - 1.0, min=-1.0, max=1.0)
        ys = torch.clamp(2.0 * ys - 1.0, min=-1.0, max=1.0)
        grids = torch.stack([xs, ys], dim=-1).flatten(0, 1)  ## Bxnq, s1, s2, 2
        x = x.unsqueeze(1).repeat_interleave(nq, dim=0).flatten(0, 1)  ## Bxnq, C, H, W
        return F.grid_sample(x, grid=grids, mode="bilinear").unflatten(0, (B, nq))  ## B, nq, C, s1, s2

@SALIENCY_INSTANCE_SEG.register()
class MultiQ(nn.Module):
    @configurable
    def __init__(self, num_queries=100, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=3, key_features=["res5","res4","res3"], mask_key="res2", grid_sizes=[(2,2),(3,3),(4,4)]):
        super().__init__()
        self.q = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.qpe = nn.Parameter(torch.randn((1, num_queries, embed_dim)))

        # self.pe_layer = PositionEmbeddingSine(embed_dim//2, normalize=True)
        assert len(key_features) == len(grid_sizes), f"{key_features} length of neq {grid_sizes}"
        self.grid_embeddings = nn.ParameterDict(dict(
            (key, nn.Parameter(torch.randn(tuple(grid_size)+(embed_dim,))))  ## s1,s2,C
            for key, grid_size in zip(key_features, grid_sizes)
        ))
        self.multiq_layers = nn.ModuleList([
            SeeThroughMultiQ(embed_dim=embed_dim, mlp_layers=2)
            for _ in range(num_blocks)
        ])
        self.self_attn_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_attn=dropout_attn) for _ in
             range(num_blocks)])
        self.cross_attn_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_attn=dropout_attn) for _ in
             range(num_blocks)])
        self.ffn_layers = nn.ModuleList(
            [FFNLayer(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout_ffn=dropout_ffn) for _ in
             range(num_blocks)])
        self.global_self_attn_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout_attn=dropout_attn) for _ in
             range(num_blocks)])
        self.global_ffn_layers = nn.ModuleList(
            [FFNLayer(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout_ffn=dropout_ffn) for _ in
             range(num_blocks)])

        self.level_embed = nn.Embedding(len(key_features), embedding_dim=embed_dim)
        self.key_features = key_features
        self.mask_key = mask_key
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_queries = num_queries

        self.head = Head(embed_dim=embed_dim, num_heads=num_heads)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_queries":  cfg.MODEL.COMMON.NUM_QUERIES,
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks":   cfg.MODEL.SIS_HEAD.NUM_BLOCKS,
            "key_features":   cfg.MODEL.SIS_HEAD.KEY_FEATURES,
            "mask_key":       cfg.MODEL.SIS_HEAD.MASK_KEY,
            "grid_sizes":     cfg.MODEL.MODULES.MULTIQ.GRID_SIZES
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
        mask_feature = feats[self.mask_key]

        B, C, H, W = mask_feature.shape
        nq = self.num_queries
        n_keys = len(self.key_features)
        sizes = [feats[key].shape[2::] for key in self.key_features]

        q = self.q.expand(B, -1, -1)
        qpe = self.qpe.expand(B, -1, -1)

        predictions = [self.head(q=q, mask_feature=mask_feature, attn_size=sizes[0])]
        for idx in range(self.num_blocks):
            key = self.key_features[int(idx % n_keys)]
            attn_mask = predictions[-1]["attn_mask"]
            bboxes = predictions[-1]["bboxes"].sigmoid()

            # split: q, qpe, z, zpe, bboxes, grid_embedding
            q_split = self.multiq_layers[idx](
                q=q, 
                qpe=qpe, 
                feat=feats[key] + self.level_embed.weight[idx % n_keys][None, :, None, None], 
                feat_pe=feats_pe[key], 
                bboxes=bboxes, 
                grid_embedding=self.grid_embeddings[key]
            ).reshape(B*nq, C, -1).transpose(-1, -2)  ## B*nq, s*s, C

            # Local Self-Attn: query, memory, memory_mask, pos = None, query_pos = None
            q_split = self.self_attn_layers[idx](
                query=q_split,
                memory=q_split,
                memory_mask=None,
                pos=None,
                query_pos=None
            )  ## B*nq, s*s, C

            # aggregate: query, memory, memory_mask, pos = None, query_pos = None
            q = self.cross_attn_layers[idx](
                query=q.flatten(0, 1).unsqueeze(1),  ## Bxnq, 1, C
                memory=q_split,
                memory_mask=None,
                pos=None,
                query_pos=None
            ).reshape(B, nq, C)  ## B, nq, C

            ## local FFN
            q = self.ffn_layers[idx](q)
            
            # Gloabl Self-Attn: query, memory, memory_mask, pos = None, query_pos = None
            q = self.global_self_attn_layers[idx](
                query=q,
                memory=q,
                memory_mask=None,
                pos=qpe,
                query_pos=qpe
            )

            # Global FFN
            q = self.global_ffn_layers[idx](q)

            predictions.append(self.head(q=q, mask_feature=mask_feature, attn_size=sizes[(idx+1) % n_keys]))

        out = predictions[-1]
        aux = predictions[0:-1]
        return q, qpe, out, aux
