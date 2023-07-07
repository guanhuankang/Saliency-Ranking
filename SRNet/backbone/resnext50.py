import torch.nn as nn
import torchvision.models as models
from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register
class ResNeXt50(nn.Module):
    @configurable
    def __init__(self, cfg):
        super().__init__()
        resnext50 = models.resnext50_32x4d(pretrained=True)
        self.layer0 = nn.Sequential(
            resnext50.conv1,
            resnext50.bn1,
            resnext50.relu,
            resnext50.maxpool
        )
        self.layer1 = resnext50.layer1
        self.layer2 = resnext50.layer2
        self.layer3 = resnext50.layer3
        self.layer4 = resnext50.layer4

        self.key_features = cfg.MODEL.BACKBONE.FEATURE_KEYS

    def forward(self, x):
        out = {}
        out["res1"] = self.layer0(x)
        out["res2"] = self.layer1(out["res1"])
        out["res3"] = self.layer2(out["res2"])
        out["res4"] = self.layer3(out["res3"])
        out["res5"] = self.layer4(out["res4"])
        ret = dict((k, out[k]) for k in out if k in self.key_features)
        return ret
