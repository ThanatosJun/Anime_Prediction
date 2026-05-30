"""
RAG Cross Attention Module

Q：MetaData Projection 輸出        [batch, 1, 128]
K/V：RAG 3 種模態投影後 concat      [batch, 15, 128]
     - rag_meta  [batch, 5, 10]   → Linear(10,   128)
     - rag_text  [batch, 5, 768]  → Linear(768,  128)
     - rag_image [batch, 5, 1024] → Linear(1024, 128)
Output：                           [batch, 128]
"""

import torch
import torch.nn as nn


class ProjectionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RAGCrossAttention(nn.Module):
    """
    Args:
        proj_dim   : projection 維度（預設 128）
        num_heads  : attention head 數（預設 4）
        dropout    : projection block dropout
        attn_dropout: attention dropout
    """

    def __init__(
        self,
        proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()

        # RAG 三種模態的 Projection Block
        self.rag_meta_proj  = ProjectionBlock(10,   proj_dim, dropout)
        self.rag_text_proj  = ProjectionBlock(768,  proj_dim, dropout)
        self.rag_image_proj = ProjectionBlock(1024, proj_dim, dropout)

        # Multi-Head Cross Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(proj_dim)

    def forward(
        self,
        query: torch.Tensor,          # [batch, 128]（MetaData Projection 輸出）
        rag_meta: torch.Tensor,       # [batch, 5, 10]
        rag_text: torch.Tensor,       # [batch, 5, 768]
        rag_image: torch.Tensor,      # [batch, 5, 1024]
        rag_mask: torch.Tensor = None,# [batch, 5] bool，True = padding（無效位置）
    ) -> torch.Tensor:                # [batch, 128]

        # 三種模態各自投影到 proj_dim
        meta_kv  = self.rag_meta_proj(rag_meta)    # [batch, 5, 128]
        text_kv  = self.rag_text_proj(rag_text)    # [batch, 5, 128]
        image_kv = self.rag_image_proj(rag_image)  # [batch, 5, 128]

        # concat 成 K/V → [batch, 15, 128]
        kv = torch.cat([meta_kv, text_kv, image_kv], dim=1)

        # Q: [batch, 128] → [batch, 1, 128]
        q = query.unsqueeze(1)

        # key_padding_mask: [batch, 15]（若有 mask，重複 3 次對應 3 種模態）
        key_padding_mask = None
        if rag_mask is not None:
            key_padding_mask = rag_mask.repeat(1, 3)  # [batch, 15]

        # Cross Attention
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        # attn_out: [batch, 1, 128]

        out = self.norm(attn_out.squeeze(1))  # [batch, 128]
        return out
