"""
FusionMLP: modality-aware fusion MLP for anime popularity / score prediction.

Architecture:
  text_emb  (384) ──→ text_proj  (Linear → LayerNorm → GELU) → text_proj  ──→ × α_t   ─┐
  image_emb (1024) ─→ image_proj (Linear → LayerNorm → GELU) → image_proj ──→ × α_img  ─┤→ concat(fused_dim) → backbone → head
  meta_rag  (65)  ──→ meta_proj  (Linear → LayerNorm → GELU) → meta_proj  ──→ × α_meta ─┘

  fused_dim = text_proj + image_proj + meta_proj  (e.g. 128+64+64=256)

  Modality gate (independent per modality, sees only its own projection):
    α_t   = softmax( [Linear(text_proj→1)(t), Linear(image_proj→1)(img), Linear(meta_proj→1)(m)] )[0]
    α_img = softmax( ... )[1]
    α_meta= softmax( ... )[2]

Backbone: Dropout → [Linear → LayerNorm → GELU → Dropout] × N → Linear(1)
"""
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _proj_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


class FusionMLP(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        meta_dim: int,
        text_proj: int = 128,
        image_proj: int = 256,
        meta_proj: int = 64,
        hidden_dims: List[int] = None,
        dropout: float = 0.4,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # per-modality projection
        self.text_proj  = _proj_block(text_dim,  text_proj)
        self.image_proj = _proj_block(image_dim, image_proj)
        self.meta_proj  = _proj_block(meta_dim,  meta_proj)

        # modality gate: each gate sees only its own projection → semantic guarantee
        self.text_gate  = nn.Linear(text_proj,  1)
        self.image_gate = nn.Linear(image_proj, 1)
        self.meta_gate  = nn.Linear(meta_proj,  1)

        fused_dim = text_proj + image_proj + meta_proj

        # backbone
        layers: List[nn.Module] = [nn.Dropout(dropout)]
        in_dim = fused_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(in_dim, 1)

        self._cfg = {
            "text_dim": text_dim, "image_dim": image_dim, "meta_dim": meta_dim,
            "text_proj": text_proj, "image_proj": image_proj, "meta_proj": meta_proj,
            "hidden_dims": hidden_dims, "dropout": dropout,
        }

    def forward(
        self,
        text: torch.Tensor,   # (B, text_dim)  — GNN-enhanced text embedding
        image: torch.Tensor,  # (B, image_dim) — GNN-enhanced image embedding
        meta: torch.Tensor,   # (B, meta_dim)
    ) -> torch.Tensor:
        t   = self.text_proj(text)
        img = self.image_proj(image)
        m   = self.meta_proj(meta)

        # modality gate: each scalar from its own projection, softmax across 3
        gates = F.softmax(
            torch.cat([self.text_gate(t), self.image_gate(img), self.meta_gate(m)], dim=1),
            dim=1,
        )  # (B, 3)

        fused = torch.cat([
            gates[:, 0:1] * t,
            gates[:, 1:2] * img,
            gates[:, 2:3] * m,
        ], dim=1)  # (B, fused_dim)

        return self.head(self.backbone(fused)).squeeze(-1)

    def get_gates(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        """Return gate weights (B, 3): [α_text, α_image, α_meta]."""
        self.eval()
        with torch.no_grad():
            t   = self.text_proj(text)
            img = self.image_proj(image)
            m   = self.meta_proj(meta)
            return F.softmax(
                torch.cat([self.text_gate(t), self.image_gate(img), self.meta_gate(m)], dim=1),
                dim=1,
            )

    def get_config(self) -> dict:
        return dict(self._cfg)

    def save_config(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._cfg, f, indent=2)

    @classmethod
    def from_config(cls, config_path: str) -> "FusionMLP":
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(**cfg)

    @classmethod
    def load(cls, config_path: str, checkpoint_path: str, map_location=None) -> "FusionMLP":
        """Reconstruct FusionMLP from saved config + checkpoint.

        Handles both checkpoint formats:
          - new: {"fusion_mlp": ..., "text_gnn": ..., "image_gnn": ...}
          - old: state_dict directly (backward compat)
        """
        model = cls.from_config(config_path)
        ckpt  = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        state = ckpt["fusion_mlp"] if isinstance(ckpt, dict) and "fusion_mlp" in ckpt else ckpt
        model.load_state_dict(state)
        return model
