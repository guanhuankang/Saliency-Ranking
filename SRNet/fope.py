import os, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .neck import FrcPN
from .decoder import Mask2FormerDecoder
from .head import MaskDecoder
from .loss import hungarianMatcher, batch_mask_loss
from .utils import calc_iou, debugDump


@META_ARCH_REGISTRY.register()
class FovealPeripheralNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.neck = FrcPN(cfg)
        self.decoder = Mask2FormerDecoder(cfg)
        self.head = MaskDecoder(cfg)

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict, *args, **argw):
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std

        feats = self.backbone(images)
        feats = self.neck(feats)
        torch.cuda.empty_cache()

        query = self.decoder(feats, deep_supervision=False)
        feat = feats["res3"]
        zpe = self.decoder.get_dense_pe(size=feat.shape[2::], b=len(feat)).flatten(2).transpose(-1, -2)
        qpe = self.decoder.get_query_pos(b=len(feat))
        pred_masks, iou_scores, obj_scores = self.head(query, feat, qpe, zpe)
        torch.cuda.empty_cache()

        if self.training:
            B = len(batch_dict)
            H, W = batch_dict[0]["masks"].shape[-2::]  ## target size
            masks = [x["masks"].to(self.device) for x in batch_dict]  ## targets
            pred_masks = F.interpolate(pred_masks, size=(H, W), mode="bilinear")  ## resize
            indices = hungarianMatcher(preds={"masks": pred_masks, "scores": obj_scores}, targets=masks)  ## list of tuples of indices

            stack_pred_masks = torch.cat([pred_masks[b,indices[b][0],:,:] for b in range(B)], dim=0)  ## K,H,W
            stack_tgt_masks = torch.cat([masks[b][indices[b][1]] for b in range(B)], dim=0)  ## K,H,W
            stack_iou = torch.cat([iou_scores[b,indices[b][0],0] for b in range(B)], dim=0)  ## K
            stack_iou_gt = torch.stack([calc_iou(p.sigmoid().detach(), m) for p, m in zip(stack_pred_masks, stack_tgt_masks)])  ## K
            pos = F.binary_cross_entropy_with_logits(obj_scores, torch.ones_like(obj_scores), reduction="none")
            neg = F.binary_cross_entropy_with_logits(obj_scores, torch.zeros_like(obj_scores), reduction="none")
            mat = torch.zeros_like(obj_scores)
            for b in range(B):
                mat[b, indices[b][0], :] += 1.0

            mask_loss = batch_mask_loss(stack_pred_masks, stack_tgt_masks).mean()
            obj_loss = (pos * mat).sum() / (mat.sum() + 1e-6) + 0.1 * (neg * (1.0 - mat)).sum() / ((1.0 - mat).sum() + 1e-6)
            iou_loss = ((torch.sigmoid(stack_iou)-stack_iou_gt)**2).mean()

            if np.random.rand() < 0.2:
                k = 5
                plst = list(stack_pred_masks[0:k].detach().float().cpu().sigmoid())
                gtlst = list(stack_tgt_masks[0:k].detach().float().cpu())
                ioulst = stack_iou[0:k].detach().float().cpu().sigmoid().tolist()
                iougt = stack_iou_gt[0:k].detach().float().cpu().tolist()
                debugDump(
                    output_dir=self.cfg.OUTPUT_DIR,
                    image_name="latest",
                    texts=[ioulst, iougt],
                    lsts=[plst, gtlst],
                    size=(256, 256)
                )

            return {
                "mask_loss": mask_loss * 10.0,
                "obj_loss": obj_loss * 2.0,
                "iou_loss": iou_loss * 2.0
            }

        else:
            ## Inference
            results = []
            for b in range(len(batch_dict)):
                H, W = batch_dict[b]["height"], batch_dict[b]["width"]
                image_name = batch_dict[b].get("image_name") or "unknown_{}".format(b)
                is_obj = torch.where(obj_scores[b] > .0)[0]
                masks = F.interpolate(pred_masks[b, is_obj].float().unsqueeze(0), size=(H, W), mode="bilinear").sigmoid().cpu()  ## 1,k,H,W
                scores = iou_scores[b, is_obj, 0].sigmoid().float().cpu()  ## k
                results.append({"image_name": image_name, "masks": list(masks[0]), "scores": scores.tolist(), "num": len(scores)})
            return results
