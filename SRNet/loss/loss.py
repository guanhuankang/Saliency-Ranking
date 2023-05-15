
import torch
import torch.nn.functional as F

def batch_mask_loss(preds, targets):
    """
    CE loss + dice loss

    Args:
        preds: B,* logits
        targets: B,* binary
    Returns:
        loss: B,1
    """
    ce_loss_weight = 1.0
    dice_loss_weight = 1.0
    preds = preds.flatten(1)  ## B,-1
    targets = targets.flatten(1)  ## B, -1
    sig_preds = torch.sigmoid(preds)  ## B, -1

    ce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none").mean(dim=-1)  ## B
    dice_loss = 1.0 - ((2. * sig_preds * targets).sum(dim=-1)+1.0) / ((sig_preds+targets).sum(dim=-1) + 1.0)  ## B
    return ce_loss * ce_loss_weight + dice_loss * dice_loss_weight

def batch_cls_loss(preds, targets):
    return F.binary_cross_entropy_with_logits(preds, targets, reduction="none").flatten(1).mean(dim=-1)