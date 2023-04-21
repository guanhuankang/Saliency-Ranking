import torch
import torch.nn as nn

class IOREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_d2 = nn.Seq

    def forward(self, ior_mask):
        '''
        ior mask embedding
        :param ior_mask: R^{B,1,H,W}
        :return: [R^{B,nh,H/4,W/4}, R^{B,nh,H/8,W/8}, R^{B,nh,H/16,W/16}, R^{B,nh,H/32,W/32}]
        '''
