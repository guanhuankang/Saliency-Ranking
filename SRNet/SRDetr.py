import os, cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from .decoder import SRDetrDecoder, Neck
from .loss import calc_mask_loss_with_score_loss, objectness_loss

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

    def debugDump(self, image_name, text, lst, size=(256, 256)):
        """
        Args:
            lst: list of image H, W
        """
        os.makedirs(self.cfg.OUTPUT_DEBUG, exist_ok=True)
        lst = [cv2.resize((x.cpu().detach().numpy()*255).astype(np.uint8), size) for x in lst]
        out = Image.fromarray(np.concatenate(lst, axis=1))
        ImageDraw.Draw(out).text((0, 0), text, fill="red")
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
            masks = [x["masks"].to(self.device) for x in batch_dict]
            selection = [np.random.randint(tot) for tot in [len(m) for m in masks]]
            targets = torch.stack([m[s] for s, m in zip(selection, masks)], dim=0).unsqueeze(1).to(self.device)  ## B, 1, H, W
            ior_masks = [m[0:s].to(self.device) for s, m in zip(selection, masks)]  ## list of torch.Tensor

            pred_masks, ior_scores = self.decoder(feat=feat, ior_masks=ior_masks)

            general_masks = tuple(pred_masks[:,0:-1,:,:])  ## Tuple B, nq-1, H, W
            general_scores = tuple(ior_scores[:,0:-1, :])  ## Tuple B, nq-1, 1
            general_mask_loss = sum([objectness_loss(pred_masks=general_masks[i], ref_masks=masks[i], pred_iou=general_scores[i]) for i in range(B)])
            target_mask_loss = calc_mask_loss_with_score_loss(pred=pred_masks[:, -1::, :, :], target=targets, pred_iou=ior_scores[:, -1::, :])
            
            if np.random.rand() < 0.1:
                self.debugDump(
                    image_name="latest",
                    text = f"score: {torch.sigmoid(ior_scores[0, -1].float().cpu().detach())}",
                    lst = [torch.sigmoid(pred_masks[0, -1]), targets[0, 0]],
                    size = (256, 256)
                )
                self.debugDump(
                    image_name="queries",
                    text = f"score: {torch.sigmoid(ior_scores[0, 0:-1].float().cpu().detach())}",
                    lst = tuple(torch.sigmoid(pred_masks[0, 0:-1])),
                    size = (48, 48)
                )
            

            return {
                "general_mask_loss": general_mask_loss,
                "target_mask_loss": target_mask_loss
            }

        else:
            ## Inference
            results = []
            for b_i in range(len(batch_dict)):
                H, W = batch_dict[b_i]["height"], batch_dict[b_i]["width"]
                ior_masks = batch_dict[b_i].get("ior_masks") or torch.zeros((0,)+images.shape[-2::], device=self.device)
                utmost_objects = batch_dict[b_i].get("utmost_objects") or self.cfg.TEST.UTMOST_OBJECTS
                image_name = batch_dict[b_i].get("image_name") or "unknown_{}".format(b_i)

                masks = []
                scores = []
                while utmost_objects > 0:
                    utmost_objects -= 1
                    mask, score = self.decoder(feat=feat[b_i:b_i+1, :, :, :], ior_masks=[ior_masks])  ## 1,nq,4H,4W | 1,nq,1
                    mask = F.interpolate(mask, size=(H, W), mode="bilinear")[0, -1].cpu() ## resize back to original size: H, W
                    score = score[0, -1].cpu()  ## torch.float
                    masks.append(torch.sigmoid(mask))
                    scores.append(torch.sigmoid(score))
                results.append({"image_name": image_name, "masks": masks, "scores": scores, "num": len(scores)})
            return results
