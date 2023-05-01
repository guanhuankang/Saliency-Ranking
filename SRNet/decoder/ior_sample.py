import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from SRNet.component.samplepoints import PointSampler

class IORSample(nn.Module):
    def __init__(self, num_points=512, embed_dim=256):
        super().__init__()
        self.num_points = num_points
        self.point_sampler = PointSampler()
        self.cls_embedding = nn.Embedding(2, embedding_dim=embed_dim)

    def sample(self, x, ior_masks):
        _, C, H, W = x.shape
        num_ior = len(ior_masks)
        n = self.num_points // num_ior
        r = self.num_points % num_ior
        ior_masks = [F.interpolate(m.unsqueeze(0).unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True) for m in ior_masks]
        points = []
        indices = []
        for i in range(num_ior):
            num = (n+1) if i < r else n
            p, idx = self.point_sampler.samplePoints2D(x, num, mask=ior_masks[i].gt(.5), indices=True)
            points.append(p)
            indices.append(torch.stack(idx, dim=0))
        points = torch.cat(points, dim=1) ## 1,k,C
        indices = torch.cat(indices, dim=1) ## 3,* (0,H~,W~)

        points = points + self.cls_embedding(torch.zeros((1, self.num_points), dtype=torch.long))
        return points, indices

    def mock(self, x):
        _, C, H, W = x.shape
        points = self.cls_embedding(torch.ones((1, self.num_points), dtype=torch.long)) ## 1,k,C
        hs = torch.randint(low=0, high=H, size=(1, self.num_points), dtype=torch.long)
        ws = torch.randint(low=0, high=W, size=(1, self.num_points), dtype=torch.long)
        indices = torch.cat([torch.zeros_like(hs), hs, ws], dim=0)
        return points, indices

    def forward(self, feat, IOR_masks, IOR_ranks=None):
        points = []
        indices = []
        for i in range(len(feat)):
            if len(IOR_masks[i]>0):
                ps, idx = self.sample(feat[i:i+1], IOR_masks[i])
            else:
                ps, idx = self.mock(feat[i:i+1])
            points.append(ps)
            idx[0, :] = idx[0, :] + i
            indices.append(idx)
        points = torch.cat(points, dim=0)
        indices = torch.cat(indices, dim=1)
        return points, tuple(indices)



