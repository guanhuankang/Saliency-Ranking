import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from ..component import Attention, MLPBlock, init_weights_

class AttnFFN(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=2048, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_ffn)
        self.norm2 = nn.LayerNorm(embed_dim)

        init_weights_(self)
    
    def forward(self, q, k, v):
        q = self.norm1(q + self.dropout1(self.attn(q=q, k=k, v=v)))
        q = self.norm2(q + self.dropout2(self.mlp(q)))
        return q

class KNetStep(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=2048, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.fc1 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.fc2 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.fc3 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.fc4 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))

        self.interaction = AttnFFN(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            dropout_attn=dropout_attn, 
            dropout_ffn=dropout_ffn
        )

        init_weights_(self.linear1)
        init_weights_(self.linear2)
        init_weights_(self.fc1)
        init_weights_(self.fc2)
        init_weights_(self.fc3)
        init_weights_(self.fc4)
        
    def forward(self, q, xq):
        """
        q: B, nq, C
        xq: B, C, h, w
        """
        Fg = self.linear1(xq) * self.linear2(q)  ## B, nq, C
        Gk = torch.sigmoid(self.fc1(Fg))
        Gf = torch.sigmoid(self.fc2(Fg))
        q = Gf * self.fc3(xq) + Gk * self.fc4(q)
        return self.interaction(q=q, k=q, v=q)

class FovealStep(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=2048, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        FIX_DIM = 16
        assert embed_dim % FIX_DIM == 0, "{} != 0".format(embed_dim % FIX_DIM)
        dims = (embed_dim // FIX_DIM, FIX_DIM)

        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(dims[0] * embed_dim))
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(embed_dim * dims[1]))
        )
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        
        init_weights_(self)

        self.interaction = AttnFFN(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            dropout_attn=dropout_attn, 
            dropout_ffn=dropout_ffn
        )

        self.sizes = ((dims[0], embed_dim), (embed_dim, dims[1]))
        
    def forward(self, q, xq):
        """
        q: B, nq, C
        xq: B, nq, C
        """
        q_key = self.mlp1(q).unflatten(-1, self.sizes[0])
        q_value = self.mlp2(xq).unflatten(-1, self.sizes[1])
        q = self.fc( torch.relu(q_key @ q_value).flatten(2) )  ## B, nq, C
        return self.interaction(q=q, k=q, v=q)  ## B, nq, C

class FovealDynamic(nn.Module):
    @configurable
    def __init__(self, mergeName="KNetStep", num_queries=100, embed_dim=256, num_heads=8, hidden_dim=2048, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=3, key_features=["res5","res4","res3"]):
        super().__init__()
        self.q = nn.Parameter(torch.zeros((1, num_queries, embed_dim)))
        self.qpe = nn.Parameter(torch.randn((1, num_queries, embed_dim)))

        self.layers = nn.ModuleDict(dict((f"{key}-{nb}", {"FovealStep": FovealStep, "KNetStep": KNetStep}[mergeName](
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout_attn=dropout_attn,
            dropout_ffn=dropout_ffn
        )) for nb in range(num_blocks) for key in key_features))

        self.conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.num_blocks = num_blocks
        self.keys = key_features
        self.level_embed = nn.ParameterDict(dict((key, nn.Parameter(torch.randn(embed_dim))) for key in key_features))

        self.mask_embed = nn.Conv2d(embed_dim, embed_dim, 1)
        self.bbox_head = nn.Linear(embed_dim, 4)
        self.score_head = nn.Linear(embed_dim, 1)
    
    def forward(self, feats, feats_pe):
        """
        Args:
            feats, feats_pe: dict of B,C,Hi,Wi
        """
        high_res_key = self.keys[-1]
        mask_features = feats[high_res_key]

        q = self.q.expand(len(mask_features), -1, -1)
        qpe = self.qpe.expand(len(mask_features), -1, -1)

        predictions = [self.predict_head(q=q, mask_features=mask_features)]
        for nb in range(self.num_blocks):
            for key in self.keys:
                x = feats[key] + self.level_embed[key][None, :, None, None]  ## B, C, h, w
                x = self.conv(x + feats_pe[key])  ## B, C, h, w
                masks = self.getMask(predictions[-1], size=x.shape[2::])  ## B, nq, h, w
                xq = masks.flatten(2) @ x.flatten(2).transpose(-1, -2)  ## B, nq, C
                q = self.layers[f"{key}-{nb}"](q=q, xq=xq)  ## B, nq, C
                predictions.append(self.predict_head(q=q, mask_features=mask_features))

        aux_predictions = predictions[0:-1]
        out = predictions[-1]
        return q, qpe, out, aux_predictions


    def predict_head(self, q, mask_features):
        """
        Args:
            q: B, nq, C
            mask_features: B, C, H, W
        Return:
            masks: B, nq, *size [logits]
            bboxes: B, nq, 4 [logits]
            scores: B, nq, 1 [logits]
        """
        H, W = mask_features.shape[2::]
        masks = q @ self.mask_embed(mask_features).flatten(2)
        masks = masks.unflatten(-1, (H, W))
        bboxes = self.bbox_head(q)
        scores = self.score_head(q)
        return {"masks": masks, "bboxes": bboxes, "scores": scores}
    
    def getMask(self, prediction, size):
        """
        Args:
            prediction: dict with field "masks"
        Return:
            masks: B, nq, *size [0,1]
        """
        ## sigmoid, bool?, detach?
        return F.interpolate(prediction["masks"], size=size, mode="bilinear").sigmoid().detach()

    @classmethod
    def from_config(cls, cfg):
        return {
            "mergeName":    cfg.MODEL.MODULES.FOVEALDYNAMIC.MERGE_NAME,
            "num_queries":  cfg.MODEL.COMMON.NUM_QUERIES,
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks":   cfg.MODEL.MODULES.FOVEALDYNAMIC.NUM_BLOCKS,
            "key_features":   cfg.MODEL.MODULES.FOVEALDYNAMIC.KEY_FEATURES
        }
