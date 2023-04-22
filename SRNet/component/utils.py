import torch.nn as nn

def init_weights_(m):
    for w in m.modules():
        if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
            nn.init.xavier_normal_(w.weight)
            nn.init.zeros_(w.bias)
        elif isinstance(w, nn.LayerNorm):
            nn.init.constant_(w.weight, 1.0)
            nn.init.zeros_(w.bias)