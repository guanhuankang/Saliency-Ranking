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
from .utils import calc_iou, debugDump, pad1d


@META_ARCH_REGISTRY.register()
class PERP(nn.Module):
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
        if self.training:
            masks = [x["masks"].to(self.device) for x in batch_dict]  ## list of k_i, Ht, Wt
            n_max = max([len(x) for x in masks])
            train_size = tuple(masks[0].shape[-2::])  ## (Ht, Wt)
            r1 = torch.tensor([[i, np.argmax(x["ranks"])] for i, x in enumerate(batch_dict)])
            r1 = r1.to(self.device).to(torch.long)

        bs = len(batch_dict)
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std

        z = self.backbone(images)
        z = self.neck(z)
        q, z, q_pe, z_pe, size = self.decoder(z)
        q, z, p_obj, p_sal = self.gsv(q=q, z=z, q_pe=q_pe, z_pe=z_pe)

        nq = q.shape[1]
        tk = min(n_max + n_max, nq) if self.training else (p_obj.gt(0.0).sum(dim=1)).max()
        oi1 = torch.arange(bs, device=self.device, dtype=torch.long).repeat_interleave(tk)
        oi2 = p_obj.topk(tk, dim=1)[1].view(-1)  ## B * tk
        p_obj_tk = p_obj[oi1, oi2, :].reshape(bs, tk, 1)  ## B, tk, 1
        p_sal_tk = p_sal[oi1, oi2, :].reshape(bs, tk, 1)  ## B, tk, 1

        q = q[oi1, oi2, :].reshape(bs, tk, -1)  ## B, tk, C
        q_pe = q_pe[oi1, oi2, :].reshape(bs, tk, -1)  ## B, tk, C
        q, p_mask, p_iou = self.fv(q=q, z=z, q_pe=q_pe, z_pe=z_pe, size=size, pe_layer=self.decoder)

        """
        q: query_warp, B, tk, tk, C
        p_obj, p_sal: B, nq, 1
        p_mask, p_iou: B, tk, (H,W)/1
        p_obj_tk: B, tk, 1
        p_sal_tk: B, tk, 1
        bs, nq, tk: int
        """
        if self.training:
            """ Training """
            p_mask = F.interpolate(p_mask, size=train_size, mode="bilinear")
            bi, qi, ti = hungarianMatcher(preds={"masks": p_mask, "scores": p_obj_tk}, targets=masks)

            """ GT
            gt_obj, gt_sal: B, nq, 1
            stack_p_mask, stack_gt_mask: K, Ht, Wt
            stack_p_iou, stack_gt_iou: K
            """
            r1_diff = (torch.stack([bi, ti], dim=1).unsqueeze(1) - r1.unsqueeze(0)).abs().sum(dim=-1)  ## a,B,[\2]
            ri = torch.unique(torch.where(r1_diff.eq(0))[0])

            ## p_sal GT
            gt_sal_tk = torch.zeros_like(p_obj_tk)
            gt_sal_tk[bi[ri], qi[ri]] = 1.0
            gt_sal = torch.zeros_like(p_sal)
            gt_sal[oi1, oi2, :] = gt_sal_tk.flatten(0, 1)

            ## p_obj GT
            gt_obj_tk = torch.zeros_like(p_obj_tk)
            gt_obj_tk[bi, qi] = 1.0
            gt_obj = torch.zeros_like(p_obj)
            gt_obj[oi1, oi2, :] = gt_obj_tk.flatten(0, 1)

            ## stack p_mask,p_iou GT
            gt_mask = torch.stack([pad1d(x, 0, n_max) for x in masks], dim=0)  ## B, n_max, Ht, Wt
            stack_p_mask = p_mask[bi, qi]  ## K, Ht, Wt
            stack_gt_mask = gt_mask[bi, ti]  ## K, Ht, Wt
            stack_p_iou = p_iou[bi, qi, 0]  ## K
            stack_gt_iou = torch.tensor(
                [calc_iou(p, g) for p, g in zip(list(stack_p_mask.sigmoid()), list(stack_gt_mask))],
                device=self.device
            )

            """ Peripheral Inhibition and Selection (q_m)
            ti: 0 is most salient, 1 is second, ...
            Aij = r_i/(r_j), if Aij<1|Aij=1|Aij>1 then r_i<r_j|r_i==r_j|r_i>r_j, 
                we have no_ihb (0), inh (1), inh (1)
            Bij = 1 means r_i+1=r_j (selection gt) 
            q_m: B, tk, tk
            
            p_sel: B, tk, tk logit
            gt_sel: B, tk, tk, binary{0,1}
            stack_p_sel: K,tk logit
            stack_gt_sel: K,tk binary
            """
            v = torch.ones((bs, tk, 1), device=self.device) * (n_max + 10)
            v[bi, qi, 0] = ti + 1.0
            A = v @ (1.0 / v).transpose(-1, -2)  ## v \times 1/v, B, tk, tk
            B = (v + 1) @ (1.0 / v).transpose(-1, -2)  ## v+1 \times 1/v, B, tk, tk
            q_m = A.ge(1.0).float()
            p_sel = self.ps(q_w=q, q_m=q_m)
            gt_sel = B.isclose(torch.tensor(1.0, dtype=B.dtype)).float()
            assert gt_sel.sum(dim=-1).max() <= 1.0005, "{} > 1.0".format(gt_sel.sum(dim=-1).max())
            stack_p_sel = p_sel[bi, qi]  ## K, tk
            stack_gt_sel = gt_sel[bi, qi]  ## K, tk
            sel_masking = stack_gt_sel.sum(dim=-1).gt(0.0).float()

            """ Loss
            p_obj (gt_obj), p_sal (gt_sal): B, nq, 1 [BCE, softmax]
            stack_p_mask, stack_gt_mask: K, Ht, Wt   [BCE+DICE]
            stack_p_iou, stack_gt_iou: K      [MSE]
            stack_p_sel, stack_gt_sel: K, tk  [softmax]
            """
            obj_pos_weight = torch.tensor(self.cfg.LOSS.WEIGHTS.OBJ_POS, device=self.device)
            obj_loss = F.binary_cross_entropy_with_logits(p_obj, gt_obj, pos_weight=obj_pos_weight)
            sal_loss = F.cross_entropy(p_sal.squeeze(-1), torch.argmax(gt_sal.squeeze(-1), dim=1))
            mask_loss = batch_mask_loss(preds=stack_p_mask, targets=stack_gt_mask).mean()
            iou_loss = ((stack_p_iou.sigmoid() - stack_gt_iou) ** 2).mean()
            sel_loss = (F.cross_entropy(stack_p_sel, torch.argmax(stack_gt_sel, dim=-1), reduction="none") *
                        sel_masking).sum() / (sel_masking.sum()+1e-6)

            """ Debug """
            if np.random.rand() < 0.2:
                k = 5
                plst = list(stack_p_mask[0:k].detach().float().cpu().sigmoid())
                gtlst = list(stack_gt_mask[0:k].detach().float().cpu())
                ioulst = stack_p_iou[0:k].detach().float().cpu().sigmoid().tolist()
                iougt = stack_gt_iou[0:k].detach().float().cpu().tolist()
                debugDump(
                    output_dir=self.cfg.OUTPUT_DIR,
                    image_name="latest",
                    texts=[ioulst, iougt],
                    lsts=[plst, gtlst],
                    size=(256, 256)
                )

            w_obj = self.cfg.LOSS.WEIGHTS.OBJ
            w_sal = self.cfg.LOSS.WEIGHTS.SAL
            w_mask = self.cfg.LOSS.WEIGHTS.MASK
            w_iou = self.cfg.LOSS.WEIGHTS.IOU
            w_sel = self.cfg.LOSS.WEIGHTS.SEL
            return {
                "obj_loss": obj_loss * w_obj,
                "sal_loss": sal_loss * w_sal,
                "mask_loss": mask_loss * w_mask,
                "iou_loss": iou_loss * w_iou,
                "sel_loss": sel_loss * w_sel
            }

        else:
            """ Inference
            input: p_mask, p_iou, p_obj_tk, p_sal_tk, p_sel: (B, tk, [H,W]/[1]/[tk])
            output: list of dict with keys:
                image_name:
                masks: [in order] list of torch.Tensor [Ho,Wo] in [0,1]
                scores:[in order] list of float in [0,1] IOU scores
                obj_scores: list of float in [0,1]
                num: int, count of instances
            [in order]: from highest to lowest salient degree
            
            orders: index of last salient object across tk queries over batch
            overall_scores: sal/sel_score x obj_score 
            """
            p_sal_tk = p_sal_tk.softmax(dim=1).squeeze(-1)  ## B, tk
            p_obj_tk = p_obj_tk.sigmoid().squeeze(-1)  ## B, tk
            p_iou = p_iou.sigmoid().squeeze(-1)  ## B, tk

            bs_idx = torch.arange(bs, device=self.device, dtype=torch.long)  ## B
            order = torch.argmax(p_sal_tk, dim=1)  ## B
            overall_scores = [p_sal_tk[bs_idx, order] * p_obj_tk[bs_idx, order]]  ## B
            orders = [order]

            q_m = torch.diag(torch.ones(tk, device=self.device)).unsqueeze(0).expand(bs, -1, -1)  ## Eye: B, tk, tk
            for i in range(1, tk):
                nxt_scores = self.ps(
                    q_w=q[bs_idx, order].unsqueeze(1),   ## B,1,tk,C
                    q_m=q_m[bs_idx, order].unsqueeze(1)  ## B,1,tk
                ).softmax(dim=-1).squeeze(1)  ## (B,tk) prob

                """ update q_m """
                q_m[bs_idx, :, order] = 1.0

                """ update order """
                order = torch.argmax(nxt_scores, dim=-1)  ## B

                orders.append(order)
                overall_scores.append(nxt_scores[bs_idx, order] * p_obj_tk[bs_idx, order])

            """
            orders: list of 1-D tensor.Tensor [B]
            overall_scores: list of overall scores [B] 
            """
            results = []
            for i in range(bs):
                Ho, Wo = batch_dict[i]["height"], batch_dict[i]["width"]
                image_name = batch_dict[i]["image_name"]
                rank_idx = torch.tensor([o[i] for o in orders])

                oa_scores = torch.tensor([o[i] for o in overall_scores])
                pred_masks = F.interpolate(p_mask[i:i+1], size=(Ho, Wo), mode="bilinear").sigmoid()[rank_idx][0]
                scores = p_iou[i][rank_idx]  ## iou
                obj_scores = p_obj[i][rank_idx]
                num = len(obj_scores)

                toC = lambda x: x.detach().float().cpu()
                results.append({
                    "image_name": image_name,
                    "masks": list(toC(pred_masks)),
                    "scores": toC(scores).tolist(),
                    "obj_scores": toC(obj_scores).tolist(),
                    "overall_scores": toC(oa_scores).tolist(),
                    "num": num
                })
            return results
            # end inference
        # """ end if """
    # """ end forward """
# """ end class """
