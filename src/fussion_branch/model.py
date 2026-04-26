"""
FusionMLP: concat(text_emb, metadata_vec, rag_features) → MLP → scalar

Architecture (per pipeline doc):
    Linear → BatchNorm → ReLU → Dropout
    Linear → BatchNorm → ReLU → Dropout
    Linear(1)

Two separate model instances are trained — one for popularity, one for meanScore.
"""
from typing import List

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
