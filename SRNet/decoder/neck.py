import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from ..component import init_weights_, LayerNorm2D


class Neck(nn.Module):
    @configurable
    def __init__(self, dim=256, in_dims=[128, 256, 512, 1024], feat_keys=["res2", "res3", "res4", "res5"]):
        super().__init__()
        D = dict((k,v) for k, v in zip(feat_keys, in_dims))
        self.conv1 = nn.Conv2d(D["res5"], dim, 1)
        self.conv2 = nn.Conv2d(D["res4"], dim, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim+dim, dim, 3, padding=1),
            nn.ReLU(),
            LayerNorm2D(dim)
        )
        init_weights_(self)

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.NECK.DIM,
            "in_dims": cfg.MODEL.BACKBONE.NUM_FEATURES,
            "feat_keys": cfg.MODEL.BACKBONE.FEATURE_KEYS
        }

    def forward(self, x):
        return self.conv3(
            torch.cat([
                F.interpolate(self.conv1(x["res5"]), scale_factor=2, mode="bilinear"),
                self.conv2(x["res4"])
            ], dim=1)
        )
