import torch
import torch.nn.functional as F
import torchvision
from ..utils import xyxy2xyhw

def batch_mask_loss(preds, targets, cfg=None):
    """
    CE loss + dice loss

    Args:
        preds: B,* logits
        targets: B,* binary
    Returns:
        loss: B,1
    """
    if cfg is None:
        ce_loss_weight = 1.0
        dice_loss_weight = 1.0
    else:
        ce_loss_weight = cfg.LOSS.MASK_CE_COST
        dice_loss_weight = cfg.LOSS.MASK_DICE_COST
        
    preds = preds.flatten(1)  ## B,-1
    targets = targets.flatten(1)  ## B, -1
    sig_preds = torch.sigmoid(preds)  ## B, -1

    ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none").mean(dim=-1)  ## B
    dice_loss = 1.0 - ((2. * sig_preds * targets).sum(dim=-1)+1.0) / ((sig_preds+targets).sum(dim=-1) + 1.0)  ## B
    return ce_loss * ce_loss_weight + dice_loss * dice_loss_weight

# def batch_cls_loss(preds, targets):
#     return F.binary_cross_entropy_with_logits(preds, targets, reduction="none").flatten(1).mean(dim=-1)

def batch_bbox_loss(box1, box2, cfg=None):
    """
    boxes in [(x1,y1),(x2,y2)]
    Args:
        box1: N, 4 [0, 1]
        box2: N, 4 [0, 1]

    Returns:
        loss: N
    """
    if cfg is None:
        bbox_l1_weight = 1.0
        bbox_giou_weight = 1.0
    else:
        bbox_l1_weight = cfg.LOSS.BBOX_L1_COST
        bbox_giou_weight = cfg.LOSS.BBOX_GIOU_COST

    version = [int(_) for _ in torchvision.__version__.split("+")[0].split(".")]
    if version[1] >= 15:
        gloss = torchvision.ops.generalized_box_iou_loss(box1, box2)  ## N
    else:
        gloss = -torch.diag(torchvision.ops.generalized_box_iou(box1, box2))  ## N
    l1loss = F.l1_loss( xyxy2xyhw(box1), xyxy2xyhw(box2), reduction="none").mean(dim=-1)
    return l1loss * bbox_l1_weight + gloss * bbox_giou_weight


