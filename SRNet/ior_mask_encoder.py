import torch
import torch.nn as nn
from detectron2.config import configurable

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
    def __init__(self, hidden_dim=32, n_head=8):
        super().__init__()
        self.conv_list = nn.ModuleList([ 
            basicLayer(1, hidden_dim=hidden_dim, n_head=n_head, kernel_size=patch_size) 
            for patch_size in [32,16,8,4]
        ])
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.MODEL.IOR_MASK_ENCODER.HIDDEN_DIM, 
            "n_head": cfg.MODEL.IOR_MASK_ENCODER.NUM_HEAD
        }

    def forward(self, ior_mask):
        '''
        ior mask embedding
        :param ior_mask: R^{B,1,H,W}
        :return: [R^{B,nh,H/32,W/32},R^{B,nh,H/16,W/16},R^{B,nh,H/8,W/8},R^{B,nh,H/4,W/4}]
        '''
        return [ conv(ior_mask) for conv in self.conv_list ]



