import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from ..component import IORMaskEncoder
from ..component import init_weights_

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
        init_weights_(self)

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
            ffn_drop = 0.0
        ):
        ''' norm before = True '''
        super().__init__()

        self.q_w = nn.Linear(dim, dim)
        self.kv_w = nn.Linear(dim, dim + dim)
        self.ffn = Mlp(dim, ffn_dim, dim)
        self.ffn_drop = nn.Dropout(p=ffn_drop)
        self.input_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )

        self.norm_q = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        init_weights_(self)
        

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.IOR_TRANSFORMER.DIM,
            "ffn_dim": cfg.MODEL.IOR_TRANSFORMER.FFN_DIM,
            "ffn_drop": cfg.MODEL.IOR_TRANSFORMER.FFN_DROP
        }

    def forward(self, query, mem, ior_emb):
        '''
        IORBlock decodes res5/4/3 once at a time.
        @param:
            query: R^{B,1,C} saliency query
            mem: R^{B,L,C} L==H*W
            ior_emb: R^{B,nh,H,W}
        @return: query' \in R^{B,1,C}
        '''
        B_, L, C = mem.shape
        B_, nh, H, W = ior_emb.shape
        assert L == H * W, "{}!={}*{}".format(L, H, W)
        
        short_cut = query
        h_dim = C//nh
        qscale = h_dim**(-0.5)

        ior_emb = ior_emb.flatten(2).split(1, dim=1) ## B,1,L \times nh
        mem = self.input_proj(mem) ## proj & norm
        key, value = self.kv_w(mem).split(C, dim=-1)
        qs = self.q_w(self.norm_q(query)).split(h_dim, dim=-1)
        ks = torch.split(key, h_dim, dim=-1)
        vs = torch.split(value, h_dim, dim=-1)
        attns = [ (q@k.transpose(-1,-2))*qscale*ior for q,k,ior in zip(qs,ks,ior_emb) ] ## MCA
        attns = [ torch.softmax(attn, dim=-1) for attn in attns ]

        x = torch.cat([ attn@v for attn,v in zip(attns,vs)], dim=-1)
        x = x + short_cut
        x = x + self.ffn_drop(self.ffn(self.norm_ffn(x)))
        return x

class IORHead(nn.Module):
    @configurable
    def __init__(self, dim=256, hidden_dim=512, out_dim=256):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.cls = nn.Linear(dim, 1)
        
        init_weights_(self)
    
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
        B_, L, C = mem.shape
        assert L == H * W, "{}!={}*{}".format(L, H, W)
        
        mem = self.input_proj(mem)
        pred_mask = self.mlp(query)@mem.transpose(-1,-2) ## B,1,L
        pred_mask = pred_mask.reshape(B_, 1, H, W)
        score = self.cls(query).view(-1,1) ## B,1
        return pred_mask, score

class IORDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.used_levels  = cfg.MODEL.IOR_DECODER_BLOCK.USED_LEVELS
        self.ior_encoder = IORMaskEncoder(cfg)
        self.ior_transformers = nn.ModuleDict((level,IORTransformer(cfg)) for level in self.used_levels)
        self.ior_head = IORHead(cfg)
        
    def forward(self, query, ior_mask, x):
        ''' decode query to the most salient instance
            under IOR mechanism given ior_mask
            @param:
                query: R^{B,1,C} saliency query (refer to the most salient obj)
                ior_mask: R^{B,1,H,W} inhibition of return mask
                x: dict of features
            @return:
                query, salient instance (under inhibition mechanism)
        '''
        B_, C, H, W = ior_mask.shape

        ior_dict = self.ior_encoder(ior_mask)
        for level in self.used_levels:
            query = self.ior_transformers[level](
                query = query, 
                mem = x[level].flatten(2).transpose(-1,-2),
                ior_emb = ior_dict[level]
            )
        pixel_emb = F.interpolate(x[self.used_levels[-1]], size=(H,W), mode="bilinear", align_corners=False)
        pixel_emb = pixel_emb.flatten(2).transpose(-1,-2)
        pred_mask, score = self.ior_head(query, pixel_emb, H, W)
        return query, pred_mask, score

class IORDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.used_levels  = cfg.MODEL.IOR_DECODER_BLOCK.USED_LEVELS
        num_blocks = cfg.MODEL.IOR_DECODER.NUM_BLOCKS
        self.blocks = nn.ModuleList([IORDecoderBlock(cfg) for i in range(num_blocks)])
    
    def forward(self, x, ior_mask = None):
        '''
        @param:
            x: dict likes {"res2":*, "res3": *,...}
            ior_mask: R^{B,1,H,W}
        @return:
            resutls: list of dict with following fields:
                mask: logit of predicted mask
                score: saliency score (logit)
                stage: 1/2/... (stage No.)
        '''
        if ior_mask==None:
            ior_mask = torch.zeros_like(x[self.used_levels[-1]])
        collect_results = lambda p,s,t=-1: {"mask": p, "score": s, "stage": t}
        results = []
        query = torch.mean(x[self.used_levels[0]], dim=[-1,-2], keepdim=False).unsqueeze(1) ## B,1,C
        for i,block in enumerate(self.blocks):
            query, pred_mask, score = block(query, ior_mask, x)
            results.append(collect_results(pred_mask, score,i+1))
        return results

