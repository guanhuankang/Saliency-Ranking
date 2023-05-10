import os, cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from .decoder import SRDetrDecoder, Neck
from .loss import hungarianMatcher, batch_mask_loss

def calc_iou(p, t):
    mul = (p*t).sum()
    add = (p+t).sum()
    return mul / (add - mul + 1e-6)

@META_ARCH_REGISTRY.register()
class SRDetr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.neck = Neck(cfg)
        self.decoder = SRDetrDecoder(cfg)

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    def debugDump(self, image_name, texts, lsts, size=(256, 256)):
        """
        Args:
            texts: list of text
            lsts: list of list of image H, W
        """
        os.makedirs(self.cfg.OUTPUT_DEBUG, exist_ok=True)
        outs = []
        for text, lst in zip(texts, lsts):
            lst = [cv2.resize((x.numpy()*255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR) for x in lst]
            out = Image.fromarray(np.concatenate(lst, axis=1))
            ImageDraw.Draw(out).text((0, 0), str(text), fill="red")
            outs.append(np.array(out))
        out = Image.fromarray(np.concatenate(outs, axis=0))
        out.save(os.path.join(self.cfg.OUTPUT_DEBUG, image_name+".png"))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict, *args, **argw):
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std
        feat = self.neck(self.backbone(images))  ## B, C, 2H, 2W
        torch.cuda.empty_cache()

        if self.training:
            B = len(batch_dict)
            H, W = batch_dict[0]["masks"].shape[-2::]
            masks = [x["masks"].to(self.device) for x in batch_dict]
            pred_masks, iou_scores, obj_scores = self.decoder(feat=feat)
            pred_masks = F.interpolate(pred_masks, size=(H,W), mode="bilinear")

            indices = hungarianMatcher(preds={"masks": pred_masks, "scores": obj_scores}, targets=masks)  ## list of tuples of indices
            stack_pred_masks = torch.cat([pred_masks[b,indices[b][0],:,:] for b in range(B)], dim=0)  ## K,H,W
            stack_tgt_masks = torch.cat([masks[b][indices[b][1]] for b in range(B)], dim=0)  ## K,H,W
            stack_iou = torch.cat([iou_scores[b,indices[b][0],0] for b in range(B)], dim=0)  ## K
            stack_iou_gt = torch.stack([calc_iou(p.sigmoid(), m) for p, m in zip(stack_pred_masks, stack_tgt_masks)])  ## K
            pos = F.binary_cross_entropy_with_logits(obj_scores, torch.ones_like(obj_scores), reduction="none")
            neg = F.binary_cross_entropy_with_logits(obj_scores, torch.zeros_like(obj_scores), reduction="none")
            mat = torch.zeros_like(obj_scores)
            for b in range(B):
                mat[b, indices[b][0], :] += 1.0
            mask_loss = F.binary_cross_entropy_with_logits(stack_pred_masks, stack_tgt_masks)
            obj_loss = (pos * mat).mean() + 0.1 * (neg * (1.0 - mat)).mean()
            iou_loss = ((torch.sigmoid(stack_iou)-stack_iou_gt)**2).mean()

            if np.random.rand() < 0.2:
                plst = list(stack_pred_masks[0:5].detach().float().cpu().sigmoid())
                ioulst = stack_iou[0:5].detach().float().cpu().sigmoid().tolist()
                gtlst = list(stack_tgt_masks[0:5].detach().float().cpu())
                iougt = stack_iou_gt[0:5].detach().float().cpu().tolist()
                self.debugDump(
                    image_name="latest",
                    texts = [ioulst, iougt],
                    lsts = [plst, gtlst],
                    size = (200, 200)
                )

            return {
                "mask_loss": mask_loss * 5.0,
                "obj_loss": obj_loss * 2.0,
                "iou_loss": iou_loss * 2.0
            }

        else:
            torch.cuda.empty_cache()
            ## Inference
            pred_masks, iou_scores, obj_scores = self.decoder(feat=feat)
            results = []
            for b in range(len(batch_dict)):
                H, W = batch_dict[b]["height"], batch_dict[b]["width"]
                image_name = batch_dict[b].get("image_name") or "unknown_{}".format(b)
                is_obj = torch.sigmoid(obj_scores[b, :, 0]).detach().cpu() > .5
                masks = []
                scores = []
                for i, obj_ in enumerate(is_obj):
                    if obj_:
                        mask = F.interpolate(pred_masks[b:b+1, i:i+1], size=(H, W), mode="bilinear")[0, 0].detach().cpu()
                        mask = mask.gt(0.0).float()
                        masks.append(mask)
                        scores.append(iou_scores[b, i, 0].sigmoid().detach().cpu())
                results.append({"image_name": image_name, "masks": masks, "scores": scores, "num": len(scores)})
            return results
