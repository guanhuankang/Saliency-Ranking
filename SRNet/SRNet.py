import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from .decoder import IORDecoder, Neck

def calc_mask_loss(pred, target):
    pred = F.interpolate(pred, size=target.shape[2::], mode="bilinear")
    sig_pred = torch.sigmoid(pred)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    dice_loss = 1.0 - 2.0 * (sig_pred * target).mean(dim=[1, 2, 3]) / ((sig_pred + target).mean(dim=[1, 2, 3]) + 1e-6)
    return bce_loss + dice_loss.mean()

def calc_cls_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


@META_ARCH_REGISTRY.register()
class SRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.neck = Neck(cfg)
        self.decoder = IORDecoder(cfg)

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict, *args, **argw):
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std
        feat = self.neck(self.backbone(images))
        torch.cuda.empty_cache()

        if self.training:
            ior_masks = [x["ior_masks"] for x in batch_dict]
            ior_ranks = [x["ior_ranks"] for x in batch_dict]

            target = torch.stack([x["target"] for x in batch_dict], dim=0).unsqueeze(1).to(self.device)
            tgt_score = torch.tensor([x["target_rank"] for x in batch_dict]).gt(0).float().to(self.device)
            results = self.decoder(feat, ior_masks, ior_ranks)
            masks, scores = results["mask"], results["score"]

            mask_loss = calc_mask_loss(masks, target)
            cls_loss = calc_cls_loss(scores.view(-1), tgt_score.view(-1))

            return {
                "mask_loss": mask_loss,
                "cls_loss": cls_loss
            }

        else:
            ## Inference
            results = []
            for b_i in range(len(batch_dict)):
                H, W = batch_dict[b_i]["height"], batch_dict[b_i]["width"]
                ior_masks = batch_dict[b_i].get("ior_masks") or []
                ior_ranks = batch_dict[b_i].get("ior_ranks") or []
                utmost_objects = batch_dict[b_i].get("utmost_objects") or self.cfg.TEST.UTMOST_OBJECTS
                image_name = batch_dict[b_i].get("image_name") or "unknown_{}".format(b_i)

                masks = []
                scores = []
                while utmost_objects > 0:
                    ret = self.decoder(feat[b_i:b_i+1], [ior_masks], [ior_ranks])
                    mask, score = ret["mask"], ret["score"].view(-1)

                    if score > 0.5:
                        ior_mask = torch.clamp(mask[0, 0], 0.0, 1.0)
                        ior_ranks.append(ior_mask)
                        masks.append(torch.sigmoid(F.interpolate(mask, size=(H, W), mode="bilinear")[0, 0]).detach().cpu())
                        scores.append(torch.sigmoid(score.detach().cpu()))
                        utmost_objects -= 1
                    else:
                        break
                results.append({"image_name": image_name, "masks": masks, "scores": scores, "num": len(scores)})
            return results
