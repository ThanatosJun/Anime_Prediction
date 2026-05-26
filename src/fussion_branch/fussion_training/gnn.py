"""
AnimeGNN: context-aware embedding enhancement via graph attention.

Graph structure per anime:
  - Node 0   : query anime (the anime being predicted)
  - Node 1..K: retrieved anime (top-K from RAG, time-filtered)

Message passing (star topology):
  - Query node aggregates from retrieved nodes via cosine-similarity attention
  - Retrieved → retrieved edges are omitted (SKAPP-lite; saves O(K²) computation)

Temporal decay:
  Attention score = cosine_sim − exp(log_time_decay) × tanh(year_gap / time_temp)
  log_time_decay is a learnable scalar (init ≈ log(0.3)); model can suppress it if
  temporal distance is not informative.  year_gaps is optional — pass None to disable.

Output: updated query embedding, same dimension as input.

Two concrete wrappers:
  TextGNN  (emb_dim=384)
  ImageGNN (emb_dim=1024)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimeGNN(nn.Module):
    """
    Args:
        emb_dim   : embedding dimension (384 for text, 1024 for image)
        num_layers: number of message-passing rounds (default 1)
        dropout   : dropout applied after each layer transform
        time_temp : temperature τ (years) for tanh decay; tanh(gap/τ) ≈ 0.76 at gap=τ
    """

    def __init__(self, emb_dim: int, num_layers: int = 1, dropout: float = 0.1,
                 time_temp: float = 10.0):
        super().__init__()
        self.emb_dim    = emb_dim
        self.num_layers = num_layers
        self.time_temp  = time_temp

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

        # Learnable temporal-decay weight (log-space → always positive).
        # init=log(0.3): moderate penalty at start; driven toward -inf to disable.
        self.log_time_decay = nn.Parameter(torch.tensor(math.log(0.3)))

    def forward(
        self,
        query_emb: torch.Tensor,            # (B, D)
        retrieved_embs: torch.Tensor,       # (B, K, D)
        retrieved_mask: torch.BoolTensor,   # (B, K)  True = valid node
        year_gaps: torch.Tensor | None = None,  # (B, K)  query_year − neighbor_year ≥ 0
    ) -> torch.Tensor:                      # (B, D)  updated query embedding
        """
        retrieved_mask: True for real retrieved anime, False for padding zeros.
        year_gaps     : non-negative temporal distance in years; None = no decay.
        When all K slots are invalid (rag_found=False), output = query_emb unchanged.
        """
        h = query_emb  # (B, D)

        for layer in self.layers:
            # ── Cosine-similarity edge weights ──────────────────────────────
            q_norm = F.normalize(h, dim=-1).unsqueeze(1)          # (B, 1, D)
            r_norm = F.normalize(retrieved_embs, dim=-1)           # (B, K, D)
            sim    = (q_norm * r_norm).sum(dim=-1)                 # (B, K)

            # ── Temporal decay penalty ────────────────────────────────────
            if year_gaps is not None:
                # penalty ∈ [0, 1]; saturates ~1 for gap >> time_temp
                penalty    = torch.tanh(year_gaps / self.time_temp)
                time_decay = torch.exp(self.log_time_decay)        # scalar > 0
                sim        = sim - time_decay * penalty

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

    def __init__(self, num_layers: int = 1, dropout: float = 0.1, time_temp: float = 10.0):
        super().__init__(emb_dim=384, num_layers=num_layers, dropout=dropout, time_temp=time_temp)


class ImageGNN(AnimeGNN):
    """GNN for image embeddings (Swin-base pooler_output, 1024-dim)."""

    def __init__(self, num_layers: int = 1, dropout: float = 0.1, time_temp: float = 10.0):
        super().__init__(emb_dim=1024, num_layers=num_layers, dropout=dropout, time_temp=time_temp)
