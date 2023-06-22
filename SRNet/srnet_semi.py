import os, cv2, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .neck import FrcPN
from .modules import BBoxDecoder, MaskDecoder, GazeShift, Foveal, FovealQ, FovealQSA
from .component import PositionEmbeddingRandom
from .utils import calc_iou, debugDump, pad1d, mask2Boxes, xyhw2xyxy, xyxy2xyhw
from .loss import hungarianMatcher, batch_mask_loss, batch_bbox_loss

@META_ARCH_REGISTRY.register()
class SRSemi(nn.Module):
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
        self.fovealqsa = FovealQSA(cfg)
        self.gaze_shift = GazeShift(cfg)

        if self.cfg.MODEL.SEMI_SUPERVISED.ENABLE:
            self.fovealqsa_ema = copy.deepcopy(self.fovealqsa)
            self.gaze_shift_ema = copy.deepcopy(self.gaze_shift)
            self.training_iter = 0

        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def updateEMA(self):
        """
        w_n' = momentum * w_o + (1.0 - momentum) * w_n
        """
        modules = [self.fovealqsa, self.gaze_shift]
        ema_modules = [self.fovealqsa_ema, self.gaze_shift_ema]
        momentum = self.cfg.MODEL.SEMI_SUPERVISED.MOMENTUM
        
        for mo, em in zip(modules, ema_modules):
            weights = {}
            for name, param in mo.named_parameters():
                weights[name] = param.data.clone()
            for name, param in em.named_parameters():
                param.data = momentum * param.data + (1.0 - momentum) * weights[name]
                param.requires_grad_(False)
        self.training_iter += 1
        thres = 0.0
        if self.training_iter >= self.cfg.MODEL.SEMI_SUPERVISED.UNLABEL_STEP[0]:
            thres = self.cfg.MODEL.SEMI_SUPERVISED.UNLABEL_RATIO[0]
        if self.training_iter >= self.cfg.MODEL.SEMI_SUPERVISED.UNLABEL_STEP[1]:
            thres = self.cfg.MODEL.SEMI_SUPERVISED.UNLABEL_RATIO[1]
        return np.random.rand() < thres

    def comm(self, inputs):
        zs = self.backbone(inputs)
        zs = self.neck(zs)

        zs_pe = dict(
            (k, self.pe_layer(zs[k].shape[2::]).unsqueeze(0).expand(len(inputs), -1, -1, -1))
            for k in zs
        )

        return zs, zs_pe
    
    @torch.no_grad()
    def inference_gaze_shift(self, zs, zs_pe, gaze_shift_key, batch_dict, ema=False):
        if ema:
            q, qpe, pred_masks, pred_bboxes, pred_objs = self.fovealqsa_ema(
                feats=zs,
                feats_pe=zs_pe
            )
            pred_bboxes = torch.sigmoid(pred_bboxes)  ## B, nq, 4 [xyhw] in [0,1]
        else:
            q, qpe, pred_masks, pred_bboxes, pred_objs = self.fovealqsa(
                feats=zs,
                feats_pe=zs_pe
            )
            pred_bboxes = torch.sigmoid(pred_bboxes)  ## B, nq, 4 [xyhw] in [0,1]

        size = tuple(zs[gaze_shift_key].shape[2::])
        bs, nq, _ = q.shape
        bs_idx = torch.arange(bs, device=self.device, dtype=torch.long)
        z = zs[gaze_shift_key].flatten(2).transpose(-1, -2)
        zpe = self.pe_layer(size).unsqueeze(0).expand(bs, -1, -1, -1).flatten(2).transpose(-1, -2)
        q_vis = torch.zeros_like(pred_objs)

        results = [{
                "image_name": x.get("image_name", idx),
                "masks": [],
                "bboxes": [],
                "scores": [],
                "saliency": [],
                "num": 0
            } for idx,x in enumerate(batch_dict)]
        for i in range(self.cfg.TEST.MAX_OBJECTS):
            if ema:
                sal = self.gaze_shift_ema(q=q, z=z, qpe=qpe, zpe=zpe, q_vis=q_vis, bbox=pred_bboxes, size=size)
            else:
                sal = self.gaze_shift(q=q, z=z, qpe=qpe, zpe=zpe, q_vis=q_vis, bbox=pred_bboxes, size=size)
            sal_max = torch.argmax(sal[:, :, 0], dim=1).long()  ##  B
            q_vis[bs_idx, sal_max, 0] = i+1

            sal_scores = sal[bs_idx, sal_max, 0].sigmoid()  ## B
            obj_scores = pred_objs[bs_idx, sal_max, 0].sigmoid()  ## B
            the_masks  = pred_masks[bs_idx, sal_max, :, :]  ## B, H, W
            the_bboxes = xyhw2xyxy(pred_bboxes[bs_idx, sal_max, :])    ## B, 4 [xyxy]

            t_sal = 0.1 if i>0 else -1.0  ## at least 1 prediction
            t_obj = 0.5 if i>0 else -1.0  ## at least 1 prediction
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

    def forward(self, batch_dict, *args, **argw):
        ## update EMA
        mode = "supervised"
        if self.training and self.cfg.MODEL.SEMI_SUPERVISED.ENABLE:
            mode = "unsupervised" if self.updateEMA() else mode
        torch.cuda.empty_cache()

        ## prepare image
        if mode=="supervised":
            images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
            inputs = (images - self.pixel_mean) / self.pixel_std
        else:
            images = torch.stack([s["un_image"] for s in batch_dict], dim=0).to(self.device).contiguous()
            inputs = (images - self.pixel_mean) / self.pixel_std

        ## feature extraction
        zs, zs_pe = self.comm(inputs=inputs)
        gaze_shift_key = self.cfg.MODEL.MODULES.GAZE_SHIFT.KEY

        ## Pseudo label from EMA model
        if mode=="unsupervised":
            out = self.inference_gaze_shift(
                zs=zs,
                zs_pe=zs_pe,
                gaze_shift_key=gaze_shift_key,
                batch_dict=batch_dict,
                ema=True
            )
            ## Pseudo label
            masks = [torch.stack([torch.from_numpy(_) for _ in x["masks"]], dim=0).to(self.device) for x in out]
        elif self.training and mode=="supervised":
            ## GT label
            masks = [x["masks"].to(self.device) for x in batch_dict]  ## list of k_i, Ht, Wt

        if self.training:
            ## Training
            q, qpe, pred_masks, pred_bboxes, pred_objs = self.fovealqsa(
                feats=zs,
                feats_pe=zs_pe
            )
            pred_bboxes = torch.sigmoid(pred_bboxes)  ## B, nq, 4 [xyhw] in [0,1]

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
            if self.cfg.LOSS.WEIGHTS.OBJ_POS < 0.0:
                n_pos = len(ti) + 1
                n_neg = int(np.prod(q_corresponse.shape)) + 1
                obj_neg_weight = torch.tensor(n_pos/(n_pos+n_neg), device=self.device)
                obj_pos_weight = torch.tensor(n_neg/(n_pos+n_neg), device=self.device) / obj_neg_weight
            if self.cfg.LOSS.WEIGHTS.OBJ_POS >= 0.0:
                obj_pos_weight = torch.tensor(self.cfg.LOSS.WEIGHTS.OBJ_POS, device=self.device)

            mask_loss = batch_mask_loss(pred_masks[bi, qi], q_masks[bi, ti]).mean()
            obj_loss = F.binary_cross_entropy_with_logits(pred_objs, q_corresponse.gt(.5).float(), pos_weight=obj_pos_weight)
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
                "mask_loss": mask_loss,
                "obj_loss": obj_loss,
                "bbox_loss": bbox_loss,
                "sal_loss": sal_loss * self.cfg.LOSS.WEIGHTS.SALIENCY
            }
            ## end training
        else:
            ## inference
            return self.inference_gaze_shift(
                zs=zs,
                zs_pe=zs_pe,
                gaze_shift_key=gaze_shift_key,
                batch_dict=batch_dict,
                ema=False
            )
            # end inference
        # """ end if """
    # """ end forward """
# """ end class """
