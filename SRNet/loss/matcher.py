import scipy
import torch
from typing import List



def hungarianMatcher(preds: List, targets: List):
    """
        Params:
            preds: list of prediction with length=batch_size, each is a dict containing:
                "masks": torch.Tensor [B,nq,H,W] logit
                "scores": torch.Tensor B,nq,1 logit
            targets: list of targets with length=batch_size, each is a torch.Tensor
                in shape N,H,W (binary map indicates the foreground/background)
        Returns:
            list of tuples: list length = batch_size, each is a pair tuple:
                (row_indices, col_indices), len(row_indices) = min(nq, N),
                where nq -> row, N -> col
    """
    