"""
FusionMLP: modality-aware fusion MLP for anime popularity / score prediction.

Architecture:
  text_emb  (384) ──→ text_proj  (Linear → LayerNorm → GELU) → 128 ──→ × α_t   ─┐
  image_emb (1024) ─→ image_proj (Linear → LayerNorm → GELU) → 256 ──→ × α_img  ─┤→ concat(449) → backbone → head
  meta_rag  (65)  ──→ meta_proj  (Linear → LayerNorm → GELU) →  64 ──→ × α_meta ─┘

  Modality gate (independent per modality, sees only its own projection):
    α_t   = softmax( [Linear(128→1)(t),   Linear(256→1)(img), Linear(64→1)(m)] )[0]
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

        self._text_dim  = text_dim
        self._image_dim = image_dim

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self._text_dim
        s2 = self._text_dim + self._image_dim
        t   = self.text_proj(x[:, :s1])
        img = self.image_proj(x[:, s1:s2])
        m   = self.meta_proj(x[:, s2:])

        # modality gate: each scalar from its own projection, softmax across 3
        gates = F.softmax(
            torch.cat([self.text_gate(t), self.image_gate(img), self.meta_gate(m)], dim=1),
            dim=1,
        )  # (B, 3)

        fused = torch.cat([
            gates[:, 0:1] * t,
            gates[:, 1:2] * img,
            gates[:, 2:3] * m,
        ], dim=1)  # (B, 448)

        return self.head(self.backbone(fused)).squeeze(-1)

    def get_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Return gate weights (B, 3) for inspection: [α_text, α_image, α_meta]."""
        self.eval()
        with torch.no_grad():
            s1 = self._text_dim
            s2 = self._text_dim + self._image_dim
            t   = self.text_proj(x[:, :s1])
            img = self.image_proj(x[:, s1:s2])
            m   = self.meta_proj(x[:, s2:])
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
        """Reconstruct model from saved config + state dict (for inference)."""
        model = cls.from_config(config_path)
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model
