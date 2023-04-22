import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from ..component import init_weights_

class FPN(nn.Module):
    @configurable
    def __init__(self, dim=256, in_dims=[128,256,512,1024], feat_keys=["res2","res3","res4","res5"]):
        super().__init__()
        self.lateral_conv = nn.ModuleDict((key,nn.Conv2d(in_dim, dim, 1)) for key,in_dim in zip(feat_keys,in_dims))
        self.output_conv = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in in_dims[1::]])
        self.feat_keys = feat_keys
        
        init_weights_(self)

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.NECK.DIM,
            "in_dims": cfg.MODEL.BACKBONE.NUM_FEATURES,
            "feat_keys": cfg.MODEL.BACKBONE.FEATURE_KEYS
        }

    def forward(self, x):
        '''
        @param:
            x: dict likes {"res2":*,"res3":*,...}
        @return: 
            dict likes {"res2":*,"res3":*,...}
        '''
        feats = [self.lateral_conv[k](x[k]) for k in self.feat_keys][::-1]
        n = len(feats)
        for i in range(n-1):
            feats[i+1] = self.output_conv[i](
                F.interpolate(feats[i], scale_factor=2, mode="bilinear", align_corners=False) \
                    + feats[i+1]
            )
        feats = feats[::-1]
        feats = dict( (k,v) for k,v in zip(self.feat_keys,feats) )
        return feats


        