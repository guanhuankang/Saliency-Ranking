import os, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .neck import FrcPN
from .decoder import Mask2FormerDecoder
# from .head import MaskDecoder
from .modules import GlobalSalienceView, FovealVision, PeripheralSelection
from .loss import hungarianMatcher, batch_mask_loss
from .utils import calc_iou, debugDump


@META_ARCH_REGISTRY.register()
class BNDM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.neck = FrcPN(cfg)
        self.decoder = Mask2FormerDecoder(cfg)
        self.gsv = GlobalSalienceView(cfg)
        self.fv = FovealVision(cfg)
        self.ps = PeripheralSelection(cfg)

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

        size = tuple(feats["res3"].shape[2::])
        B = len(batch_dict)
        query = self.decoder(feats, deep_supervision=False)
        z = feats["res3"].flatten(2).transpose(-1, -2)  ## B, HW, C
        zpe = self.decoder.get_dense_pe(size=size, b=B).flatten(2).transpose(-1, -2)
        qpe = self.decoder.get_query_pos(b=B)
        torch.cuda.empty_cache()

        query, z, obj_scores, sal_scores = self.gsv(q=query, z=z, q_pe=qpe,
                                                    z_pe=zpe)  ## B,nq,C | B,HW,C | B,nq,1 | B,nq,1
        torch.cuda.empty_cache()

        if self.training:
            masks = [x["masks"].to(self.device) for x in batch_dict]  ## targets
            n_max = max([len(x) for x in masks])
            nq = query.shapep[1]
            obj_cnt = min(nq, 2*n_max)

            obj_indice1 = torch.arange(B, device=self.device, dtype=torch.long).repeat_interleave(obj_cnt)  ## batch_obj_indice
            obj_indice2 = obj_scores.topk(obj_cnt, dim=1)[1].reshape(-1)  ## indices for candidates, B * obj_cnt
            query = query[obj_indice1, obj_indice2, :].reshape(B, obj_cnt, -1)  ## B, oc, C
            qpe = qpe[obj_indice1, obj_indice2, :].reshape(B, obj_cnt, -1)  ## B, oc, C
            query, pred_masks, iou_scores



            query, pred_masks, iou_scores = self.fv(q=query, z=z, q_pe=qpe, z_pe=zpe,
                                                    decoder=self.decoder)  ## B,nq,nq,C | B,nq,H,W | B,nq,1
            torch.cuda.empty_cache()


            H, W = batch_dict[0]["masks"].shape[-2::]  ## target size

            pred_masks = F.interpolate(pred_masks, size=(H, W), mode="bilinear")  ## resize
            indices = hungarianMatcher(preds={"masks": pred_masks, "scores": obj_scores},
                                       targets=masks)  ## list of tuples of indices



            stack_pred_masks = torch.cat([pred_masks[b, indices[b][0], :, :] for b in range(B)], dim=0)  ## K,H,W
            stack_tgt_masks = torch.cat([masks[b][indices[b][1]] for b in range(B)], dim=0)  ## K,H,W
            stack_iou = torch.cat([iou_scores[b, indices[b][0], 0] for b in range(B)], dim=0)  ## K
            stack_iou_gt = torch.stack(
                [calc_iou(p.sigmoid().detach(), m) for p, m in zip(stack_pred_masks, stack_tgt_masks)])  ## K
            pos = F.binary_cross_entropy_with_logits(obj_scores, torch.ones_like(obj_scores), reduction="none")
            neg = F.binary_cross_entropy_with_logits(obj_scores, torch.zeros_like(obj_scores), reduction="none")
            mat = torch.zeros_like(obj_scores)
            for b in range(B):
                mat[b, indices[b][0], :] += 1.0

            mask_loss = batch_mask_loss(stack_pred_masks, stack_tgt_masks).mean()
            obj_loss = (pos * mat).sum() / (mat.sum() + 1e-6) + 0.1 * (neg * (1.0 - mat)).sum() / (
                    (1.0 - mat).sum() + 1e-6)
            iou_loss = ((torch.sigmoid(stack_iou) - stack_iou_gt) ** 2).mean()

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
                if len(is_obj) > 0:
                    masks = list(F.interpolate(pred_masks[b, is_obj].float().unsqueeze(0), size=(H, W),
                                               mode="bilinear").sigmoid().cpu()[0])  ## k,H,W
                    scores = iou_scores[b, is_obj, 0].sigmoid().float().cpu().tolist()  ## k
                    obj_scores = obj_scores[b, is_obj, 0].sigmoid().float().cpu().tolist()
                else:
                    masks, scores, obj_scores = [], [], []
                results.append({
                    "image_name": image_name,
                    "masks": masks,
                    "scores": scores,
                    "obj_scores": obj_scores,
                    "num": len(scores)
                })
            return results
