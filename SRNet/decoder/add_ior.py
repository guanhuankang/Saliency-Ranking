import torch, math
from torch import Tensor, nn
from ..component import Attention, MLPBlock, init_weights_

from detectron2.config import configurable

class InverseAttention(nn.Module):
    """
    An inversed  attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        inv_attn = -attn / math.sqrt(c_per_head)
        inv_attn = torch.softmax(inv_attn, dim=-1)

        # Get output
        out = inv_attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class AddIOR(nn.Module):
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        raise "Deprecated Error"
        super().__init__()
        self.inv_attn = InverseAttention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout1 = nn.Dropout(p=dropout_attn)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.token_self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout2 = nn.Dropout(p=dropout_attn)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_ffn)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.query_emb = nn.Embedding(1, embedding_dim=embed_dim)
        self.token_emb = nn.Embedding(1, embedding_dim=embed_dim)

        self.full_self_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout4 = nn.Dropout(p=dropout_attn)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.ffn = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout5 = nn.Dropout(p=dropout_ffn)
        self.norm5 = nn.LayerNorm(embed_dim)

        self.full_to_query_attn = Attention(embedding_dim=embed_dim, num_heads=num_heads, downsample_rate=1)
        self.dropout6 = nn.Dropout(p=dropout_attn)
        self.norm6 = nn.LayerNorm(embed_dim)

        self.proj = MLPBlock(embedding_dim=embed_dim, mlp_dim=hidden_dim)
        self.dropout7 = nn.Dropout(p=dropout_ffn)
        self.norm7 = nn.LayerNorm(embed_dim)

        init_weights_(self)

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim": cfg.MODEL.IOR_DECODER.EMBED_DIM,
            "num_heads": cfg.MODEL.IOR_DECODER.CROSSATTN.NUM_HEADS,
            "dropout_attn": cfg.MODEL.IOR_DECODER.CROSSATTN.DROPOUT,
            "hidden_dim": cfg.MODEL.IOR_DECODER.FFN.HIDDEN_DIM,
            "dropout_ffn": cfg.MODEL.IOR_DECODER.FFN.DROPOUT
        }

    def forward(self, token, query):
        '''

        Args:
            token: B, nt, C
            query: B, nq, C

        Returns:
            query: B, nq, C
        '''
        nt, nq = token.shape[1], query.shape[1]

        token = self.norm1(token + self.dropout1(self.inv_attn(q=token, k=query, v=query)))
        token = self.norm2(token + self.dropout2(self.token_self_attn(q=token, k=token, v=token)))
        token = self.norm3(token + self.dropout3(self.mlp(token)))

        token_emb = torch.repeat_interleave(self.token_emb.weight, nt, dim=0).unsqueeze(0) ## 1,nt, C
        query_emb = torch.repeat_interleave(self.query_emb.weight, nq, dim=0).unsqueeze(0) ## 1,nq, C
        embs = torch.cat([token_emb, query_emb], dim=1) ## 1, nt+nq, C
        full = torch.cat([token, query], dim=1) ## B, nt+nq, C

        full = self.norm4(full + self.dropout4(self.full_self_attn(q=full+embs, k=full+embs, v=full)))
        full = self.norm5(full + self.dropout5(self.ffn(full)))

        full = self.norm6(full + self.dropout6(self.full_to_query_attn(q=full+embs, k=query+query_emb, v=query)))
        full = self.norm7(full + self.dropout7(self.proj(full)))

        query = full[:, -nq::, :] ## B, nq, C
        return query
