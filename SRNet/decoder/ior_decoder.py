import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from ..ior_mask_encoder import IORMaskEncoder
from ..position_encoding import PositionEmbeddingSine

class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class IORTransformer(nn.Module):
    @configurable
    def __init__(self, 
            dim = 256,
            ffn_dim = 512,
            ffn_drop = 0.0,
            mem_dim = 256
        ):
        ''' norm before = True '''
        super().__init__()

        self.q_w = nn.Linear(dim, dim)
        self.kv_w = nn.Linear(dim, dim + dim)
        self.ffn = Mlp(dim, ffn_dim, dim)
        self.ffn_drop = nn.Dropout(p=ffn_drop)
        self.input_proj = nn.Sequential(
            nn.Linear(mem_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

        self.norm_q = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.IOR_TRANSFORMER.DIM,
            "ffn_dim": cfg.MODEL.IOR_TRANSFORMER.FFN_DIM,
            "ffn_drop": cfg.MODEL.IOR_TRANSFORMER.FFN_DROP
        }

    def forward(self, query, mem, ior_embs):
        '''
        IORBlock decodes res5/4/3 once at a time.
        @param:
            query: R^{B,1,C} saliency query
            mem: R^{B,L,C} L==H*W
            ior_embs: R^{B,nh,H,W}
        @return: query' \in R^{B,1,C}
        '''
        mem = self.input_proj(mem) ## proj & norm

        B_, L, C = mem.shape
        B_, nh, H, W = ior_mask.shape
        assert L == H * W, "{}!={}*{}".format(L, H, W)
        
        short_cut = query
        h_dim = C//nh
        qscale = h_dim**(-0.5)

        ior_embs = ior_embs.flatten(2).split(1, dim=1) ## B,1,L \times nh
        key, value = self.kv_w(mem).split(C, dim=-1)
        qs = self.q_w(self.norm_q(query)).split(h_dim, dim=-1)
        ks = torch.split(key, h_dim, dim=-1)
        vs = torch.split(value, h_dim, dim=-1)
        attns = [ (q@k.transpose(-1,-2))*qscale*ior for q,k,ior in zip(qs,ks,ior_embs) ] ## MCA

        x = torch.cat([ attn@v for attn,v in zip(attns,vs)], dim=-1)
        x = x + short_cut
        x = x + self.ffn_drop(self.ffn(self.norm_ffn(x)))
        return x

class IORHead(nn.Module):
    @configurable
    def __init__(self, dim=256, hidden_dim=512, out_dim=256, mem_dim=256):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(mem_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.cls = nn.Linear(dim, 1)
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.IORHEAD.DIM,
            "hidden_dim": cfg.MODEL.IORHEAD.HIDDEN_DIM,
            "out_dim": cfg.MODEL.IORHEAD.OUT_DIM
        }

    def forward(self, query, mem, H, W):
        '''
        prediction head
        @param:
            query: R^{B,1,C}
            mem: R^{B,L,C}
        @return:
            pred_mask: B,1,H,W where H * W == L
            score: confidence score \in R^B
        '''
        mem = self.input_proj(mem)
        B_, L, C = mem.shape
        assert L == H * W
        
        pred_mask = self.mlp(query)@mem.transpose(-1,-2) ## B,1,L
        pred_mask = pred_mask.reshape(B_, 1, H, W)
        score = self.cls(query).view(-1) ## B
        return pred_mask, score

class IORDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_features = cfg.MODEL.BACKBONE.NUM_FEATURES
        used_levels = cfg.MODEL.IOR_DECODER_BLOCK.USED_LEVELS

        self.ior_encoder = IORMaskEncoder(cfg)
        self.ior_transformers = nn.ModuleList([
            IORTransformer(cfg, mem_dim=num_features[i])
            for i in used_levels
        ])
        self.ior_head = IORHead(cfg)
        
    def forward(self, query, ior_mask, x):
        ''' decode query to the most salient instance
            under IOR mechanism given ior_mask
            @param:
                query: R^{B,1,C} saliency query (refer to the most salient obj)
                ior_mask: R^{B,1,H,W} inhibition of return mask
                x: list of features [res5, res4, res3, ...]
            @return:
                query, salient instance (under inhibition mechanism)
        '''
        B_, C_, H, W = x[-1].shape

        ior_embs = self.ior_encoder(ior_mask)[0:len(x)]
        for ior_emb_nh, mem, layer in zip(ior_embs, x, self.ior_transformers):
            query = layer(query, mem.flatten(2).transpose(-1,-2), ior_emb_nh)
        pred_mask, score = self.ior_head(query, x[-1].flatten(2).transpose(-1,-2), H, W)

        return query, pred_mask, score


class IORDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_blocks = cfg.MODEL.IOR_DECODER.NUM_BLOCKS
        self.layers = nn.ModuleList([IORDecoderBlock(cfg) for i in range(num_blocks)])
    
    def forward(self, x, ior_mask = None):
        '''
        @param:
            x: [res5, res4, ...]
            ior_mask: R^{B,1,H,W}
        @return:
            resutls: list of dict with following fields:
                pred_mask: logit of predicted mask
                score: saliency score (logit)
                stage: 1/2/... (stage No.)
                ior_mask: ior_mask (same as input)
        '''
        if ior_mask==None:
            ior_mask = F.interpolate(torch.zeros_like(x[0]), scale_factor=32, mode="nearest")

        collect_results = lambda p,s: {"pred_mask": p, "score": s, "ior_mask": ior_mask}
        results = []

        query = torch.mean(x[0], dim=[-1,-2], keepdim=True).transpose(-1,-2) ## B,1,C
        for i,layer in enumerate(self.layers):
            query, pred_mask, score = layer(query, ior_mask, x)
            results.append(collect_results(pred_mask, score).update({"stage":i+1}))
        
        return results

