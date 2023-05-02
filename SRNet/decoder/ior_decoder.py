import torch
import torch.nn as nn
import torch.nn.functional as F


from detectron2.config import configurable
from ..component import CrossAttn, FFN, PointSampler, init_weights_
from .ior_sample import IORSample

class IORDecoderBlock(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_ca=0.0, dropout_ffn=0.0):
        super().__init__()
        self.ca_1 = CrossAttn(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_ca)
        self.sa_1 = CrossAttn(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_ca)
        self.ffn_1 = FFN(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout_ffn)

        self.ca_2 = CrossAttn(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_ca)
        self.sa_2 = CrossAttn(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_ca)
        self.ffn_2 = FFN(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout_ffn)

        self.ca_z = CrossAttn(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_ca)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "num_heads": cfg.MODEL.IOR_DECODER.CROSSATTN.NUM_HEADS,
            "dropout_ca": cfg.MODEL.IOR_DECODER.CROSSATTN.DROPOUT,
            "hidden_dim": cfg.MODEL.IOR_DECODER.FFN.HIDDEN_DIM,
            "dropout_ffn": cfg.MODEL.IOR_DECODER.FFN.DROPOUT
        }

    def getMask(self, query, mq):
        B, L, C = query.shape
        m = torch.zeros((L, L), dtype=torch.float, device=query.device)
        m[0:L-mq, L-mq] = -1e9
        return m

    def forward(self, query, query_ape, ior_points, ior_ape, z, z_ape, mask_queries=0):
        mask = self.getMask(query, mask_queries)

        query, _ = self.ca_1(tgt=query, mem=ior_points, mask=None, pe_tgt=query_ape, pe_mem=ior_ape)
        query, _ = self.sa_1(tgt=query, mem=query, mask=mask, pe_tgt=query_ape, pe_mem=query_ape)
        query = self.ffn_1(query)

        query, _ = self.ca_2(tgt=query, mem=z, mask=None, pe_tgt=query_ape, pe_mem=z_ape)
        query, _ = self.sa_2(tgt=query, mem=query, mask=mask, pe_tgt=query_ape, pe_mem=query_ape)
        query = self.ffn_2(query)

        z, _ = self.ca_z(tgt=z, mem=query, mask=None, pe_tgt=z_ape, pe_mem=query_ape)
        return query, z


class IORPixelDecoder(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.mlp = FFN(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.cls = nn.Linear(embed_dim, 1)
        self.trans_conv1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, padding=1, output_padding=1, stride=2, dilation=1)
        self.trans_conv2 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, padding=1, output_padding=1, stride=2, dilation=1)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "hidden_dim": cfg.MODEL.IOR_DECODER.FFN.HIDDEN_DIM,
            "dropout": cfg.MODEL.IOR_DECODER.FFN.DROPOUT
        }

    def forward(self, query, feat):
        feat = self.trans_conv1(feat)
        feat = self.trans_conv2(feat)
        B, C, H, W = feat.shape
        query = self.mlp(query)
        score = self.cls(query)
        mask = (query @ feat.flatten(2)).reshape(B, -1, H, W)
        return {
            "mask": mask,
            "score": score
        }

class IORDecoder(nn.Module):
    @configurable
    def __init__(self,
                 cfg,
                 embed_dim=256,
                 ape_size=(64, 64),
                 num_ior_points=256,
                 num_tokens=256,
                 num_queries=1,
                 num_blocks=6
                 ):
        super().__init__()

        self.ape = nn.Parameter(torch.zeros((1, embed_dim, ape_size[0], ape_size[1])))
        self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.xavier_normal_(self.ape)
        nn.init.xavier_normal_(self.queries)

        self.ior_sample = IORSample(num_points=num_ior_points, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([IORDecoderBlock(cfg) for _ in range(num_blocks)])
        self.points_sampler = PointSampler()
        self.pixel_decoder = IORPixelDecoder(cfg)

        self.num_queries = num_queries
        self.num_tokens = num_tokens

    @classmethod
    def from_config(cls, cfg):
        return {
            "cfg": cfg,
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "ape_size": (cfg.MODEL.IOR_DECODER.APE.HEIGHT, cfg.MODEL.IOR_DECODER.APE.WIDTH),
            "num_ior_points": cfg.MODEL.IOR_DECODER.NUM_IOR_POINTS,
            "num_tokens": cfg.MODEL.IOR_DECODER.NUM_TOKENS,
            "num_queries": cfg.MODEL.IOR_DECODER.NUM_QUERIES,
            "num_blocks": cfg.MODEL.IOR_DECODER.NUM_BLOCKS
        }

    def forward(self, feat, IOR_masks, IOR_ranks=None):
        B, C, H, W = feat.shape
        ape = F.interpolate(self.ape, size=(H,W), mode="bicubic") ## 1, C, H, W

        ior_points, ior_indices = self.ior_sample(feat, IOR_masks, IOR_ranks) ## B, #, C | (B~,H~,W~)
        ior_ape = torch.repeat_interleave(ape, B, dim=0)[ior_indices[0], :, ior_indices[1], ior_indices[2]].reshape(ior_points.shape)

        tokens, t_indices = self.points_sampler.samplePointsRegularly2D(feat, self.num_tokens, indices=True) ## B, #, C | (H~,W~)
        tokens_ape = torch.repeat_interleave(ape, B, dim=0)[:, :, t_indices[0], t_indices[1]].transpose(-1, -2)

        latent_queries = torch.repeat_interleave(self.queries, B, dim=0)
        query = torch.cat([tokens, latent_queries], dim=1)
        query_ape = torch.cat([tokens_ape, torch.zeros_like(latent_queries)], dim=1)

        z = feat.flatten(2).permute(0, 2, 1) ## B, L, C
        z_ape = torch.repeat_interleave(ape, B, dim=0).flatten(2).permute(0, 2, 1) ## B, L, C

        for block in self.blocks:
            query, z = block(query, query_ape, ior_points, ior_ape, z, z_ape, mask_queries=self.num_queries)
        feat = z.reshape(B, C, H, W)

        results = self.pixel_decoder(query[:, self.num_tokens::, :], feat)
        return results
