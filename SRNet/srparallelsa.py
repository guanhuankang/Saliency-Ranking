import os, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .neck import FrcPN, FPN
from .modules import GazeShift, FovealParallelSA
from .component import PositionEmbeddingRandom
from .utils import calc_iou, debugDump, pad1d, mask2Boxes, xyhw2xyxy, xyxy2xyhw
from .loss import hungarianMatcher, batch_mask_loss, batch_bbox_loss

@META_ARCH_REGISTRY.register()
class SRParallelSA(nn.Module):
    """
    SRFoveal: backbone+neck+fovealq+gazeshift (foveal w/o sa)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.neck = FrcPN(cfg)
        self.pe_layer = PositionEmbeddingRandom(cfg.MODEL.COMMON.EMBED_DIM//2)
        # self.bbox_decoder = BBoxDecoder(cfg)
        # self.mask_decoder = MaskDecoder(cfg)
        self.fovealqsa = FovealParallelSA(cfg)
        self.gaze_shift = GazeShift(cfg)

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict, *args, **argw):
        torch.cuda.empty_cache()
        ## prepare image
        bs = len(batch_dict)
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std

        zs = self.backbone(images)
        zs = self.neck(zs)

        zs_pe = dict(
            (k, self.pe_layer(zs[k].shape[2::]).unsqueeze(0).expand(bs, -1, -1, -1))
            for k in zs
        )

        # q, qpe, aux_bboxes, pred_objs = self.bbox_decoder(
        #     z=zs["res5"].flatten(2).transpose(-1, -2),
        #     zpe=self.pe_layer(zs["res5"].shape[2::]).unsqueeze(0).expand(bs, -1, -1, -1).flatten(2).transpose(-1, -2)
        # )
        # aux_bboxes = torch.sigmoid(aux_bboxes)  ## xyhw

        q, qpe, pred_masks, pred_bboxes, pred_objs = self.fovealqsa(
            feats=zs,
            feats_pe=zs_pe
        )
        pred_bboxes = torch.sigmoid(pred_bboxes)  ## B, nq, 4 [xyhw] in [0,1]
        gaze_shift_key = self.cfg.MODEL.MODULES.GAZE_SHIFT.KEY

        # q, pred_masks, pred_bboxes = self.foveal(
        #     q=q,
        #     qpe=qpe,
        #     feats=zs,
        #     feats_pe=zs_pe
        # )
        # pred_bboxes = torch.sigmoid(pred_bboxes)  ## xyhw

        # pred_masks = self.mask_decoder(
        #     q=q,
        #     z=zs["res3"].flatten(2).transpose(-1, -2),
        #     qpe=qpe,
        #     zpe=self.pe_layer(zs["res3"].shape[2::]).unsqueeze(0).expand(bs, -1, -1, -1).flatten(2).transpose(-1, -2),
        #     size=tuple(zs["res3"].shape[2::])
        # )

        if self.training:
            ## Training
            masks = [x["masks"].to(self.device) for x in batch_dict]  ## list of k_i, Ht, Wt
            bboxes = [mask2Boxes(m) for m in masks]  ## list of k_i, 4[xyxy]
            n_max = max([len(x) for x in masks])
            gt_size = masks[0].shape[-2::]

            pred_masks = F.interpolate(pred_masks, size=gt_size, mode="bilinear")
            bi, qi, ti = hungarianMatcher(preds={"masks": pred_masks, "scores": pred_objs}, targets=masks)

            q_masks = torch.stack([pad1d(m, dim=0, num=n_max, value=0.0) for m in masks], dim=0)     ## B, n_max, H, W
            q_boxes = torch.stack([pad1d(bb, dim=0, num=n_max, value=0.0) for bb in bboxes], dim=0)  ## B, n_max, 4

            q_corresponse = torch.zeros_like(pred_objs)  ## B, nq, 1
            q_corresponse[bi, qi, 0] = (ti + 1).to(q_corresponse.dtype)  ## 1 to n_max

            ## mask loss
            if self.cfg.LOSS.WEIGHTS.OBJ_POS < 0.0 or self.cfg.LOSS.WEIGHTS.OBJ_NEG < 0.0:
                n_pos = len(ti) + 1
                n_neg = int(np.prod(q_corresponse.shape)) + 1
                obj_pos_weight = torch.tensor(n_neg/(n_pos+n_neg), device=self.device)
                obj_neg_weight = torch.tensor(n_pos/(n_pos+n_neg), device=self.device)
            else:
                obj_pos_weight = torch.tensor(self.cfg.LOSS.WEIGHTS.OBJ_POS, device=self.device)
                obj_neg_weight = torch.tensor(self.cfg.LOSS.WEIGHTS.OBJ_NEG, device=self.device)
            
            # pos = q_corresponse.gt(.5).float()
            # neg = q_corresponse.le(.5).float()

            mask_loss = batch_mask_loss(pred_masks[bi, qi], q_masks[bi, ti]).mean()
            
            obj_loss = F.binary_cross_entropy_with_logits(pred_objs, q_corresponse.gt(.5).float(), pos_weight=obj_pos_weight/obj_neg_weight) * obj_neg_weight
            # obj_loss = (F.l1_loss(torch.sigmoid(pred_objs), torch.ones_like(pred_objs), reduction="none") * pos * obj_pos_weight + \
            # F.l1_loss(torch.sigmoid(pred_objs), torch.zeros_like(pred_objs), reduction="none") * neg).mean()

            bbox_loss = batch_bbox_loss(xyhw2xyxy(pred_bboxes[bi, qi]), q_boxes[bi, ti]).mean()
            sal_loss = torch.zeros_like(obj_loss).mean()  ## initialize as zero

            ## saliency loss
            for i in range(n_max+1):
                # q_vis_gt = q_corresponse.gt(i).float() * torch.rand_like(q_corresponse).le(0.15).float()
                q_vis = q_corresponse * q_corresponse.le(i).float()  # + q_vis_gt
                q_ans = q_corresponse.eq(i+1).float()
                sal = self.gaze_shift(
                    q=q,
                    z=zs[gaze_shift_key].flatten(2).transpose(-1, -2),
                    qpe=qpe,
                    zpe=zs_pe[gaze_shift_key].flatten(2).transpose(-1, -2),
                    q_vis=q_vis,
                    bbox=pred_bboxes,  ## xyhw
                    size=tuple(zs[gaze_shift_key].shape[2::])
                )
                sal_loss += F.binary_cross_entropy_with_logits(sal, q_ans)

            ## debugDump
            if np.random.rand() < 0.1:
                k = 5
                mm = pred_masks[bi, qi].sigmoid()[0:k].detach().cpu()  ## k, H, W
                tt = q_masks[bi, ti].cpu()[0:k]  ## k, H, W
                ss = pred_objs[bi, qi, 0].sigmoid()[0:k].detach().cpu().tolist()  ## k
                oo = [float(calc_iou(m, t)) for m, t in zip(mm, tt)]
                debugDump(
                    output_dir=self.cfg.OUTPUT_DIR,
                    image_name="latest",
                    texts=[ss, oo],
                    lsts=[list(mm), list(tt)],
                    data=None
                )

            return {
                "mask_loss": mask_loss * self.cfg.LOSS.WEIGHTS.MASK_COST,
                "obj_loss": obj_loss * self.cfg.LOSS.WEIGHTS.CLS_COST,
                "bbox_loss": bbox_loss * self.cfg.LOSS.WEIGHTS.BBOX_COST,
                "sal_loss": sal_loss * self.cfg.LOSS.WEIGHTS.SAL_COST
            }
            ## end training
        else:
            ## inference
            size = tuple(zs[gaze_shift_key].shape[2::])
            z = zs[gaze_shift_key].flatten(2).transpose(-1, -2)
            zpe = self.pe_layer(size).unsqueeze(0).expand(bs, -1, -1, -1).flatten(2).transpose(-1, -2)
            q_vis = torch.zeros_like(pred_objs)
            bs, nq, _ = q.shape
            bs_idx = torch.arange(bs, device=self.device, dtype=torch.long)

            results = [{
                    "image_name": x.get("image_name", idx),
                    "masks": [],
                    "bboxes": [],
                    "scores": [],
                    "saliency": [],
                    "num": 0
                } for idx,x in enumerate(batch_dict)]
            for i in range(nq):
                sal = self.gaze_shift(q=q, z=z, qpe=qpe, zpe=zpe, q_vis=q_vis, bbox=pred_bboxes, size=size)
                sal_max = torch.argmax(sal[:, :, 0], dim=1).long()  ##  B
                q_vis[bs_idx, sal_max, 0] = i+1

                sal_scores = sal[bs_idx, sal_max, 0].sigmoid()  ## B
                obj_scores = pred_objs[bs_idx, sal_max, 0].sigmoid()  ## B
                the_masks  = pred_masks[bs_idx, sal_max, :, :]  ## B, H, W
                the_bboxes = xyhw2xyxy(pred_bboxes[bs_idx, sal_max, :])    ## B, 4 [xyxy]

                t_sal = 0.1
                t_obj = 0.5
                valid_items = sal_scores.gt(t_sal).float() * obj_scores.gt(t_obj).float()
                valid_idx = torch.where(valid_items.gt(.5))[0]
                for idx in valid_idx:
                    hi, wi = batch_dict[idx]["height"], batch_dict[idx]["width"]
                    results[idx]["masks"].append(
                        F.interpolate(the_masks[idx:idx+1, :, :].unsqueeze(1), size=(hi, wi), mode="bilinear")[0, 0].sigmoid().detach().cpu().gt(.5).float().numpy()
                    )
                    results[idx]["bboxes"].append(
                        (the_bboxes[idx].detach().cpu() * torch.tensor([wi, hi, wi, hi])).tolist()
                    )
                    results[idx]["scores"].append(
                        float(obj_scores[idx].detach().cpu())
                    )
                    results[idx]["saliency"].append(
                        float(sal_scores[idx].detach().cpu())
                    )
                    results[idx]["num"] += 1
                if len(valid_idx) <= 0:
                    break
            return results
            # end inference
        # """ end if """
    # """ end forward """
# """ end class """
