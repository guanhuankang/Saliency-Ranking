import torch
import torch.nn as nn

from detectron2.config import configurable
from .utils import init_weights_

class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, **argw):
        super().__init__()
        self.num_heads = num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        init_weights_(self)

    def forward(self, q, k, v, attn_mask=None):
        B, L, C = q.shape
        h_dim = C // self.num_heads
        scale = h_dim**(-0.5)

        qs = self.W_q(q).split(h_dim, dim=-1)
        ks = self.W_k(k).split(h_dim, dim=-1)
        vs = self.W_v(v).split(h_dim, dim=-1)
        attn_mask = 0.0 if isinstance(attn_mask, type(None)) else attn_mask
        attns = [
            torch.softmax((q @ k.transpose(-1, -2)) * scale + attn_mask, dim=-1)
            for q, k in zip(qs, ks)
        ]
        out = torch.cat([attn @ v for v, attn in zip(vs, attns)], dim=-1)
        return self.W_o(out), torch.stack(attns, dim=0).mean(dim=0)

class CrossAttn(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.multi_head = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        init_weights_(self)

    def with_pos_emb(self, x, pos_emb):
        if isinstance(pos_emb, type(None)):
            return x
        else:
            return x + pos_emb

    def forward(self, tgt, mem, mask=None, pe_tgt=None, pe_mem=None):
        q2 = self.with_pos_emb(self.norm_q(tgt), pe_tgt)
        k2 = self.with_pos_emb(self.norm_k(mem), pe_mem)
        q2, attn = self.multi_head(q2, k2, mem, attn_mask=mask)
        return self.dropout(q2) + tgt, attn

class FFN(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        init_weights_(self)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.act(self.linear1(self.norm(x)))))) + x

