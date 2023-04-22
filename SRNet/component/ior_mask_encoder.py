import torch
import torch.nn as nn
from detectron2.config import configurable
from .utils import init_weights_

def basicLayer(in_dim, hidden_dim, n_head, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_dim, hidden_dim, kernel_size, stride=kernel_size),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(),
        nn.Conv2d(hidden_dim, 2*hidden_dim, 1),
        nn.ReLU(),
        nn.Conv2d(2*hidden_dim, n_head, 1),
        nn.Sigmoid()
    )

class IORMaskEncoder(nn.Module):
    @configurable
    def __init__(self, hidden_dim=32, n_head=8, feature_keys=["res2","res3","res4","res5"]):
        super().__init__()
        self.conv_list = nn.ModuleList([ 
            basicLayer(1, hidden_dim=hidden_dim, n_head=n_head, kernel_size=patch_size) 
            for patch_size in [4,8,16,32]
        ])
        self.feature_keys = feature_keys
        init_weights_(self)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.MODEL.IOR_MASK_ENCODER.HIDDEN_DIM, 
            "n_head": cfg.MODEL.IOR_MASK_ENCODER.NUM_HEAD,
            "feature_keys": cfg.MODEL.IOR_MASK_ENCODER.FEATURE_KEYS
        }

    def forward(self, ior_mask):
        '''
        ior mask embedding
        :param ior_mask: R^{B,1,H,W}
        :return: dict likes
             res2: R^{B,nh,H/4,W/4},
             res3: R^{B,nh,H/8,W/8},
             res4: R^{B,nh,H/16,W/16},
             res5: R^{B,nh,H/32,W/32}
        '''
        return dict( (key,conv(ior_mask)) for key,conv in zip(self.feature_keys,self.conv_list) )

