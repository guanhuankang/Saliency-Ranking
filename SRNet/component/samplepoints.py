import torch
import torch.nn as nn
import numpy as np

class PointSampler(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

    def forward(self, x, k, mask=None, indices=True):
        return self.samplePoints2D(x, k, mask=mask, indices=indices)

    def samplePoints1D(self, x, k, mask=None, indices=True):
        B, L, C = x.shape
        if isinstance(mask, type(None)):
            mask = torch.ones_like(x[:, :, 0]) > .5  ## B,L
        if mask.dtype != torch.bool:
            mask = mask > .5
        assert mask.shape == x.shape[0:2], "mask shape {} != x.shape {}".format(mask.shape, x.shape)
        assert B > 0

        count = mask.sum(dim=1).int().cpu().detach().numpy().reshape(-1)
        cumsum = np.append(0, np.cumsum(count)[0:-1])
        assert count.min() > 0, f"mask can not be empty: {count}"

        perm_ = torch.cat([
            torch.randperm(c, device=x.device).repeat(int(np.ceil(k / c)))[0:k] + cu
            for c, cu in zip(count, cumsum)
        ]).long()

        idx_tup = torch.where(mask)
        sample_indices = tuple(idx[perm_] for idx in idx_tup)
        samples = x[sample_indices]
        samples = samples.reshape(B, k, C)
        if indices:
            return samples, sample_indices
        else:
            return samples

    def samplePoints2D(self, x, k, mask=None, indices=True):
        B, C, H, W = x.shape
        if not isinstance(mask, type(None)):
            mask = mask.flatten(2).permute(0, 2, 1).squeeze(-1)
        ret = self.samplePoints1D(x.flatten(2).permute(0, 2, 1), k, mask=mask, indices=indices)
        if indices:
            x, indices = ret
            return x, (indices[0], indices[1].div(W).long(), indices[1].remainder(W).long())
        else:
            return ret

    def samplePointsRegularly1D(self, x, k, indices=True):
        B, L, C = x.shape
        idx = np.linspace(0, L, k+2)[1:-1].astype(int)
        if indices:
            return x[:, idx, :], torch.LongTensor(idx)
        else:
            return x[:, idx, :]

    def samplePointsRegularly2D(self, x, k, indices=True):
        B, C, H, W = x.shape
        s = np.sqrt(H * W / k)
        nh = int(H // s)
        nw = int(W // s)
        remainder = int(k - nh * nw)
        ws = np.linspace(0.5*s, W-0.5*s, nw).astype(int)
        hs = np.linspace(0.5*s, H-0.5*s, nh).astype(int)
        xs, ys = np.meshgrid(hs, ws)
        xs = np.append(xs.reshape(-1), np.random.randint(0, H, remainder))
        ys = np.append(ys.reshape(-1), np.random.randint(0, W, remainder))
        assert len(xs)==len(ys) and len(xs)==k
        if indices:
            return x[:, :, xs, ys].permute(0, 2, 1), (torch.LongTensor(xs), torch.LongTensor(ys))
        else:
            return x[:, :, xs, ys].permute(0, 2, 1)

