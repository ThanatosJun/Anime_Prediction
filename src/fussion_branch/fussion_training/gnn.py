"""
AnimeGNN: context-aware embedding enhancement via graph attention.

Graph structure per anime:
  - Node 0   : query anime (the anime being predicted)
  - Node 1..K: retrieved anime (top-K from RAG, time-filtered)

Message passing (star topology):
  - Query node aggregates from retrieved nodes via cosine-similarity attention
  - Retrieved → retrieved edges are omitted (SKAPP-lite; saves O(K²) computation)

Output: updated query embedding, same dimension as input.

Two concrete wrappers:
  TextGNN  (emb_dim=384)
  ImageGNN (emb_dim=1024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimeGNN(nn.Module):
    """
    Args:
        emb_dim   : embedding dimension (384 for text, 1024 for image)
        num_layers: number of message-passing rounds (default 1)
        dropout   : dropout applied after each layer transform
    """

    def __init__(self, emb_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.emb_dim    = emb_dim
        self.num_layers = num_layers

        # Each layer: concat(query, aggregated_context) → query-dim
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(
        self,
        query_emb: torch.Tensor,            # (B, D)
        retrieved_embs: torch.Tensor,       # (B, K, D)
        retrieved_mask: torch.BoolTensor,   # (B, K)  True = valid node
    ) -> torch.Tensor:                      # (B, D)  updated query embedding
        """
        retrieved_mask: True for real retrieved anime, False for padding zeros.
        When all K slots are invalid (rag_found=False), output = query_emb unchanged.
        """
        h = query_emb  # (B, D)

        for layer in self.layers:
            # ── Cosine-similarity edge weights ──────────────────────────────
            q_norm = F.normalize(h, dim=-1).unsqueeze(1)          # (B, 1, D)
            r_norm = F.normalize(retrieved_embs, dim=-1)           # (B, K, D)
            sim    = (q_norm * r_norm).sum(dim=-1)                 # (B, K)

            # Mask padding slots before softmax
            sim = sim.masked_fill(~retrieved_mask, float("-inf"))
            attn = F.softmax(sim, dim=-1)                          # (B, K)
            # All-invalid rows → NaN after softmax; replace with 0
            attn = torch.nan_to_num(attn, nan=0.0)

            # ── Aggregate ────────────────────────────────────────────────────
            context = (attn.unsqueeze(-1) * retrieved_embs).sum(dim=1)  # (B, D)

            # ── Update: residual + transform ─────────────────────────────────
            h = h + layer(torch.cat([h, context], dim=-1))

        return self.out_norm(h)


class TextGNN(AnimeGNN):
    """GNN for text embeddings (all-MiniLM-L6-v2, 384-dim)."""

    def __init__(self, num_layers: int = 1, dropout: float = 0.1):
        super().__init__(emb_dim=384, num_layers=num_layers, dropout=dropout)


class ImageGNN(AnimeGNN):
    """GNN for image embeddings (Swin-base pooler_output, 1024-dim)."""

    def __init__(self, num_layers: int = 1, dropout: float = 0.1):
        super().__init__(emb_dim=1024, num_layers=num_layers, dropout=dropout)
