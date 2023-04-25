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

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1,-1,1,1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1,3,1,1), False)
    
    @property
    def device(self):
        return self.pixel_mean.device
    
    def mergeBatch(self, batch_dict):
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std
        ior_masks = torch.stack([s.get("ior_mask", torch.zeros_like(images[0, 0])) for s in batch_dict]).unsqueeze(1)
        return {
            "images": images,
            "ior_masks": ior_masks.to(self.device).contiguous(),
        }

    def extract_features(self, x):
        return self.necknet(self.backbone(x))

    def forward(self, batch_dict, *args, **argw):
        inp = self.mergeBatch(batch_dict)

        if self.training:
            feats = self.extract_features(inp["images"])
            results = self.decoder(feats, inp["ior_masks"])

            masks = torch.stack([s["mask"] for s in batch_dict], dim=0).unsqueeze(1).to(self.device)
            scores = torch.tensor([s["score"] for s in batch_dict]).to(self.device)
            mask_loss = sum([
                F.binary_cross_entropy_with_logits(r["mask"], masks) * w
                for r,w in zip(results, self.cfg.MODEL.IOR_DECODER.LOSS_WEIGHTS)
            ])
            cls_loss = sum([
                F.binary_cross_entropy_with_logits(r["score"], scores) * w
                for r,w in zip(results, self.cfg.MODEL.IOR_DECODER.LOSS_WEIGHTS)
            ])
            return {
                "mask_loss": mask_loss,
                "cls_loss": cls_loss
            }
        else:
            ## Inference
            feats = self.extract_features(inp["images"])

            results = []
            for b_i in range(len(batch_dict)):
                ior_mask = inp["ior_masks"][b_i:b_i+1]
                feat = dict( (k,v[b_i:b_i+1]) for k,v in feats.items() )
                utmost_objects = batch_dict[b_i].get("utmost_objects") or self.cfg.TEST.UTMOST_OBJECTS
                H, W = batch_dict[b_i]["height"], batch_dict[b_i]["width"]
                image_name = batch_dict[b_i].get("image_name") or "unknown_{}".format(b_i)

                masks = []
                scores = []
                while utmost_objects>0:
                    stage = self.decoder(feat, ior_mask)[-1] ## get last stage
                    score = torch.sigmoid(stage["score"]).view(-1) ## scalar
                    mask = torch.sigmoid(stage["mask"]) ## 1,1,~,~
                    if score > 0.5:
                        ior_mask = torch.clamp(ior_mask + mask, 0.0, 1.0)
                        masks.append(F.interpolate(mask, size=(H,W), mode="bilinear")[0,0])
                        scores.append(score)
                        utmost_objects -= 1
                    else:
                        break
                results.append({"image_name": image_name, "masks": masks, "scores": scores, "num": len(scores)})
            return results

