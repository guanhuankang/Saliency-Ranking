import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align
from detectron2.config import configurable
# from ..component import PositionEmbeddingSine
from .registry import SALIENCY_INSTANCE_SEG
from ..utils import xyhw2xyxy
from ..component import Attention, MLPBlock, init_weights_

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

class ROISample(nn.Module):
    def __init__(self, embed_dim=256, roi_windows=(4,4)):
        super().__init__()
        self.roi_windows = roi_windows
        self.grid_embedding = nn.Parameter(torch.randn([embed_dim, roi_windows[0], roi_windows[1]]))
    
    def forward(self, feat, boxes):
        """
        Args:
            feat: B, C, H, W
            boxes: B, nq, 4 [xyhw in [0,1]]
        Return:
            roi_feat: B, nq, C, h, w
        """
        B, _, H, W = feat.shape
        _, nq, _ = boxes.shape

        b_index = torch.arange(B, device=boxes.device).reshape(B, 1, 1).expand(B, nq, 1)
        boxes = torch.clamp(xyhw2xyxy(boxes), min=0.0, max=1.0)  ## B, nq, 4
        boxes = boxes * torch.tensor([[[W, H, W, H]]], device=boxes.device)  ## B, nq, 4
        boxes = torch.cat([b_index, boxes], dim=-1).flatten(0, 1)  ## Bxnq, 5

        roi_feat = roi_align(feat, boxes, output_size=self.roi_windows)  ## Bxnq, C, *roi_windows
        roi_feat = roi_feat + self.grid_embedding[None, :, :, :]
        roi_feat = roi_feat.unflatten(0, (B, nq))  ## B, nq, C, h, w

        return roi_feat

class LookByMultiQ(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.read_out = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))

        self.cross_attn1 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn2 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.self_attn1 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout3 = nn.Dropout(p=dropout_attn)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.self_attn2 = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout4 = nn.Dropout(p=dropout_attn)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.ffn1 = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout5 = nn.Dropout(p=dropout_ffn)
        self.norm5 = nn.LayerNorm(embed_dim)

        self.ffn2 = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout6 = nn.Dropout(p=dropout_ffn)
        self.norm6 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def forward(self, q, qpe, feat, feat_pe, roi_feat):
        """
        q, qpe: B, nq, C
        feat, feat_pe: B, C, H, W
        roi_feat: B, nq, C, h, w
        """
        B, nq, _, h, w = roi_feat.shape

        copyq = self.linear1(q).unsqueeze(2).repeat_interleave(h*w, dim=2)  ## B, nq, h*w, C
        roi_feat = self.linear2(roi_feat.flatten(3).transpose(-1, -2))  ## B, nq, h*w, C
        multi_q = (copyq + roi_feat).flatten(1, 2)  ## B, nq*h*w, C
        multi_qpe = qpe.repeat_interleave(h*w, dim=1)  ## B, nq*hw, C
        
        z = feat.flatten(2).transpose(-1, -2)
        zpe = feat_pe.flatten(2).transpose(-1, -2)

        ## multi_q branch
        multi_q = self.norm1(multi_q + self.dropout1(self.cross_attn1(q=multi_q+multi_qpe, k=z+zpe, v=z)))
        multi_q = multi_q.reshape(B*nq, h*w, -1)  ## B*nq, h*w, C
        multi_q = self.norm2(multi_q + self.dropout2(self.self_attn1(q=multi_q, k=multi_q, v=multi_q)))
        multi_q = self.norm3(multi_q + self.ffn1(multi_q))  ## B*nq, h*w, C
        multi_q = multi_q.mean(dim=1).unflatten(0, (B, nq))  ## B, nq, C

        ## global branch
        q = self.norm4(q + self.dropout4(self.cross_attn2(q=q+qpe, k=z+zpe, v=z)))
        q = self.norm5(q + self.dropout5(self.self_attn2(q=q+qpe, k=q+qpe, v=q)))
        q = self.norm6(q + self.dropout6(self.ffn2(q)))  ## B, nq, C
        
        ## read out
        q = self.read_out(q + multi_q)  ## B, nq, C
        return q

@SALIENCY_INSTANCE_SEG.register()
class MultiQ(nn.Module):
    @configurable
    def __init__(self, num_queries=100, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=3, key_features=["res5","res4","res3"], mask_key="res2", grid_sizes=[(2,2),(3,3),(4,4)]):
        super().__init__()
        assert len(key_features)==len(grid_sizes), f"{key_features} #len neq# {grid_sizes}"

        self.q = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.qpe = nn.Parameter(torch.randn((1, num_queries, embed_dim)))

        self.roi_layers = nn.ModuleList([ROISample(embed_dim=embed_dim, roi_windows=gs) for gs in grid_sizes])
        self.level_embed = nn.Embedding(len(key_features), embedding_dim=embed_dim)
        self.lookByMultiQ_layers = nn.ModuleList([
            LookByMultiQ(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn
            )
            for _ in range(num_blocks)
        ])

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
        n_keys = len(self.key_features)
        
        q = self.q.expand(len(mask_feature), -1, -1)  ## B, nq, C
        qpe = self.qpe.expand(len(mask_feature), -1, -1)  ## B, nq, C

        out_size = mask_feature.shape[2::]
        predictions = [self.head(q=q, mask_feature=mask_feature, attn_size=out_size)]
        for idx in range(self.num_blocks):
            level_emb = self.level_embed.weight[idx % n_keys].reshape(1, -1, 1, 1)
            feat = feats[self.key_features[idx % n_keys]] + level_emb
            feat_pe = feats_pe[self.key_features[idx % n_keys]]
            boxes = predictions[-1]["bboxes"].sigmoid()  ## B, nq, 4

            roi_feat = self.roi_layers[idx % n_keys](feat=feat, boxes=boxes)  ## B, nq, C, h, w
            q = self.lookByMultiQ_layers[idx](q, qpe, feat, feat_pe, roi_feat)
            
            predictions.append(self.head(q=q, mask_feature=mask_feature, attn_size=out_size))

        out = predictions[-1]
        aux = predictions[0:-1]
        return q, qpe, out, aux
