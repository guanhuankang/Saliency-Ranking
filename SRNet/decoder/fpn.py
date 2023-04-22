import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable

class FPN(nn.Module):
    @configurable
    def __init__(self, dim=256, in_dims=[256,512,1024,2048]):
        super().__init__()
        self.lateral_conv = nn.ModuleList([ nn.Conv2d(in_dim, dim, 1) for in_dim in in_dims])
        self.output_conv = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in in_dims[1::]])
        
        self.initialize()

    def initialize(self):
        for i in range(len(self.lateral_conv)):
            weight_init.c2_xavier_fill(self.lateral_conv[i])
        for i in range(len(self.output_conv)):
            weight_init.c2_xavier_fill(self.output_conv[i])

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.FPN.DIM,
            "in_dims": cfg.MODEL.BACKBONE.NUM_FEATURES
        }

    def forward(self, x):
        '''
        @param:
            x: [..., res3, res4, res5]
        @return: [..., res3, res4, res5] after top-down fusion
        '''
        assert len(x) == len(self.lateral_conv), "{}!={}".format(len(x), len(self.lateral_conv))
        n = len(x)
        x = [ self.lateral_conv[i](x[i]) for i in range(n) ][::-1] ## [res5, res4, ...]
        for i in range(len(self.output_conv)):
            x[i+1] = self.output_conv[i](x[i] + x[i+1])
        return x

