
import torch
import torch.nn.functional as F

def calc_mask_loss(pred, target):
    '''

    Args:
        pred: B, N, H, W
        target: B, N, H, W

    Returns:
        loss: torch.float

    '''
    pred = F.interpolate(pred, size=target.shape[2::], mode="bilinear")
    sig_pred = torch.sigmoid(pred)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    dice_loss = 1.0 - 2.0 * (sig_pred * target).mean(dim=[1, 2, 3]) / ((sig_pred + target).mean(dim=[1, 2, 3]) + 1e-6)
    return bce_loss + dice_loss.mean()


def calc_mask_loss_with_score_loss(pred, target, pred_iou):
    '''

    Args:
        pred: B, N, H, W logit
        target: B, N, H, W
        pred_iou: B, N, 1 logit

    Returns:
        loss: torch.float

    '''
    pred = F.interpolate(pred, size=target.shape[2::], mode="bilinear")
    sig_pred = torch.sigmoid(pred)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    iou = (sig_pred * target).mean(dim=[-1, -2]) / ((sig_pred + target).mean(dim=[-1, -2]) + 1e-6)  ## B, N
    dice_loss = (1.0 - 2.0 * iou).mean()
    score_loss = ((torch.sigmoid(pred_iou) - iou.detach().unsqueeze(-1))**2).mean()
    return bce_loss + dice_loss + score_loss * 0.4

def objectness_loss(pred_masks, ref_masks, pred_iou):
    """
    ref_masks: [r1, r2, ..., rN]
    pred_masks: [p1, p2, ..., pP]
    where P > N
    find the lowest loss to match all ref_masks (IN ORDER) from pred_masks
    that is, match r1 at first and then r2, and r3, ...

    Args:
        pred_masks: P, H, W
        ref_masks: N, H, W
        pred_iou: P, 1

    Returns:
        loss = Loss(r1, p_i) + Loss(r2, p_j) + ... + L(rN, p_k)
    """
    P, _, _ = pred_masks.shape
    N, H, W = ref_masks.shape
    pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=(H, W), mode="bilinear")[0]  ## P, H, W

    refs = torch.repeat_interleave(ref_masks, P, dim=0)  ## PxN, H, W
    preds = pred_masks.repeat((N, 1, 1))  ## PxN, H, W

    mse = torch.mean((refs - torch.sigmoid(preds))**2, dim=[1, 2])  ## MSE errors PxN
    mse = mse.reshape(N, P)  ## N, P
    argsort = torch.argsort(mse).cpu().detach() ## N, P
    ref_indexs = []
    pred_indexs = []
    for i in range(N):
        for j in argsort[i]:
            if j not in pred_indexs:
                pred_indexs.append(int(j))
                ref_indexs.append(i)
                break
    return calc_mask_loss_with_score_loss(
        pred=pred_masks[pred_indexs].unsqueeze(0),  ## 1, N, H, W
        target=ref_masks[ref_indexs].unsqueeze(0),   ## 1, N, H, W
        pred_iou=pred_iou[pred_indexs].unsqueeze(0)   ## 1, N, 1
    )


