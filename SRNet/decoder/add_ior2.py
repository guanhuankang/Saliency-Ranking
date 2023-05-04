import torch, math
from torch import Tensor, nn
from ..component import Attention, MLPBlock, init_weights_

from detectron2.config import configurable


class AddIORLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()
        self.token_to_query_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.query_to_token_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout3 = nn.Dropout(p=dropout_attn)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout4 = nn.Dropout(p=dropout_ffn)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, q, t, q_pe, t_pe):
        t = self.norm1(t + self.dropout1(self.token_to_query_attn(q=t + t_pe, k=q + q_pe, v=q)))
        q = self.norm2(q - self.dropout2(self.query_to_token_attn(q=q + q_pe, k=t + t_pe, v=t)))  ## NOTE: minus!!!
        tq = torch.cat([t, q], dim=1)
        tq_pe = torch.cat([t_pe, q_pe], dim=1)
        tq = self.norm3(tq + self.dropout3(self.self_attn(q=tq + tq_pe, k=tq + tq_pe, v=tq)))
        tq = self.norm4(tq + self.dropout4(self.mlp(tq)))

        nt, nq = t.shape[1], q.shape[1]
        t = tq[:, 0:nt, :]  ## B, nt, C
        q = tq[:, nt::, :]  ## B, nq, C
        return q, t


class AddIOR(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0, num_blocks=4):
        super().__init__()
        self.query_emb = nn.Embedding(1, embedding_dim=embed_dim)
        self.token_emb = nn.Embedding(1, embedding_dim=embed_dim)
        self.layers = nn.ModuleList([
            AddIORLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout_attn=dropout_attn,
                        dropout_ffn=dropout_ffn)
            for _ in range(num_blocks)
        ])
        self.q_to_query_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.norm = nn.LayerNorm(embed_dim)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "num_heads": cfg.MODEL.IOR_DECODER.CROSSATTN.NUM_HEADS,
            "dropout_attn": cfg.MODEL.IOR_DECODER.CROSSATTN.DROPOUT,
            "hidden_dim": cfg.MODEL.IOR_DECODER.FFN.HIDDEN_DIM,
            "dropout_ffn": cfg.MODEL.IOR_DECODER.FFN.DROPOUT,
            "num_blocks": cfg.MODEL.IOR_DECODER.ADDIOR.NUM_BLOCKS
        }

    def forward(self, token, query, token_pos, query_pos):
        '''

        Args:
            token: B, nt, C
            query: B, nq, C
            token_pos: B, nt, C
            query_pos: B, nq, C

        Returns:
            query: B, nq, C
        '''
        t_pe = token_pos + self.token_emb.weight.unsqueeze(0)  ## B, nt, C
        q_pe = query_pos + self.query_emb.weight.unsqueeze(0)  ## B, nq, C
        
        q = query
        t = token
        for layer in self.layers:
            q, t = layer(q=q, t=t, q_pe=q_pe, t_pe=t_pe)

        q = self.norm(q + self.q_to_query_attn(q=q + q_pe, k=query + q_pe, v=query))  ## B, nq, C

        return q  ## B, nq, C
