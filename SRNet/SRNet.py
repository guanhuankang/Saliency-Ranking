import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from .decoder import IORDecoder, FPN

@META_ARCH_REGISTRY.register()
class SRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.necknet = FPN(cfg)
        self.decoder = IORDecoder(cfg)

        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1,3,1,1)
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1,3,1,1)
    
    @property
    def device(self):
        return self.pixel_mean.device
    
    def extract_train(self, batch_dict):
        images = torch.stack([torch.from_numpy(s["image"]).float() for s in batch_dict], dim=0).permute(0,3,1,2).to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std
        ior_masks = torch.stack([torch.from_numpy(s["ior_mask"]).float() for s in batch_dict], dim=0).unsqueeze(1)
        masks = torch.stack([torch.from_numpy(s["mask"]).float() for s in batch_dict], dim=0).unsqueeze(1)
        scores = torch.tensor([s["score"] for s in batch_dict]).reshape(-1,1)

        return {
            "images": images,
            "ior_masks": ior_masks.to(self.device),
            "masks": masks.to(self.device),
            "scores": scores.to(self.device)
        }

    def forward(self, batch_dict, *args, **argw):
        data = self.extract_train(batch_dict)
        print(data["scores"].device)
        exit(0)
