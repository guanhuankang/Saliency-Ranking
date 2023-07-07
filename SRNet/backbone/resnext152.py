import torch.nn as nn
import torchvision.models as models
from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register
class ResNet152(nn.Module):
    @configurable
    def __init__(self, cfg):
        super().__init__()
        resnet152 = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet152.conv1,
            resnet152.bn1,
            resnet152.relu,
            resnet152.maxpool
        )
        self.layer1 = resnet152.layer1
        self.layer2 = resnet152.layer2
        self.layer3 = resnet152.layer3
        self.layer4 = resnet152.layer4

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
