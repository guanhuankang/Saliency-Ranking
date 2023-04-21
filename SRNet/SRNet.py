import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class SRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv2d(3, 1, 3)
    
    def forward(self, list_dict, *args, **argw):
        print(type(list_dict))
        pass
