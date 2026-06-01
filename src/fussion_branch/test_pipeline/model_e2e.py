"""
FusionMLPE2E: same fusion architecture as FusionMLP but with a trainable
HuggingFace text encoder backbone instead of pre-computed frozen embeddings.

Architecture:
  description ──→ [AutoModel backbone (top layers trainable)] ──→ mean-pool ──→ text_emb (384)
  image_emb   ──→ image_proj ──→ × α_img  ─┐
  text_emb    ──→ text_proj  ──→ × α_text  ─┤→ concat → backbone MLP → head
  meta_feat   ──→ meta_proj  ──→ × α_meta  ─┘

Layer freezing:
  freeze_layers=4 means embeddings + transformer layers 0–3 are frozen;
  layers 4–5 (top 2) are trainable with backbone_lr (much lower than head lr).
"""
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _proj_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over token dimension."""
    mask = attention_mask.unsqueeze(-1).float()           # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)              # (B, H)
    count  = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
    return summed / count                                  # (B, H)


def _freeze_encoder_layers(model: nn.Module, n_freeze: int) -> None:
    """Freeze token embeddings + bottom n_freeze transformer layers."""
    for param in model.embeddings.parameters():
        param.requires_grad = False
    for i in range(n_freeze):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = False


class FusionMLPE2E(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        freeze_layers: int,
        image_dim: int,
        meta_dim: int,
        text_proj: int = 128,
        image_proj: int = 64,
        meta_proj: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.4,
        no_image: bool = False,
        no_meta: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self._freeze_layers = freeze_layers
        self._no_image = no_image
        self._no_meta  = no_meta

        # ── text encoder backbone ──────────────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(encoder_name)
        text_out_dim = self.text_encoder.config.hidden_size  # 384 for MiniLM-L6
        _freeze_encoder_layers(self.text_encoder, freeze_layers)

        n_trainable_enc = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        n_total_enc     = sum(p.numel() for p in self.text_encoder.parameters())
        print(f"  [encoder] trainable {n_trainable_enc:,} / {n_total_enc:,} params "
              f"(freeze_layers={freeze_layers})")

        # ── per-modality projection ────────────────────────────────────────────
        self.text_proj = _proj_block(text_out_dim, text_proj)
        if not no_meta:
            self.meta_proj = _proj_block(meta_dim, meta_proj)
        if not no_image:
            self.image_proj = _proj_block(image_dim, image_proj)

        # ── modality gate (only active modalities) ─────────────────────────────
        n_active = 1 + (0 if no_meta else 1) + (0 if no_image else 1)
        self.text_gate = nn.Linear(text_proj, 1)
        if not no_meta:
            self.meta_gate = nn.Linear(meta_proj, 1)
        if not no_image:
            self.image_gate = nn.Linear(image_proj, 1)

        fused_dim = (text_proj
                     + (0 if no_meta  else meta_proj)
                     + (0 if no_image else image_proj))

        # ── backbone MLP ───────────────────────────────────────────────────────
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
            "encoder_name": encoder_name, "freeze_layers": freeze_layers,
            "image_dim": image_dim, "meta_dim": meta_dim,
            "text_proj": text_proj, "image_proj": image_proj, "meta_proj": meta_proj,
            "hidden_dims": hidden_dims, "dropout": dropout,
            "no_image": no_image, "no_meta": no_meta,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_emb: torch.Tensor,
        meta_feat: torch.Tensor,
    ) -> torch.Tensor:
        enc_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t = self.text_proj(_mean_pool(enc_out.last_hidden_state, attention_mask))

        parts_proj  = [t]
        parts_gate  = [self.text_gate(t)]
        if not self._no_image:
            img = self.image_proj(image_emb)
            parts_proj.append(img)
            parts_gate.append(self.image_gate(img))
        if not self._no_meta:
            m = self.meta_proj(meta_feat)
            parts_proj.append(m)
            parts_gate.append(self.meta_gate(m))

        if len(parts_gate) == 1:
            # single modality — no gate needed
            fused = parts_proj[0]
        else:
            gates = F.softmax(torch.cat(parts_gate, dim=1), dim=1)
            fused = torch.cat([gates[:, i:i+1] * p
                               for i, p in enumerate(parts_proj)], dim=1)

        return self.head(self.backbone(fused)).squeeze(-1)

    def backbone_parameters(self) -> List[nn.Parameter]:
        """Trainable encoder params — use lower LR."""
        return [p for p in self.text_encoder.parameters() if p.requires_grad]

    def head_parameters(self) -> List[nn.Parameter]:
        """All non-encoder params (projections, gates, MLP, head)."""
        enc_ids = {id(p) for p in self.text_encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc_ids]

    def get_config(self) -> dict:
        return dict(self._cfg)

    def save_config(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._cfg, f, indent=2)

    @classmethod
    def from_config(cls, config_path: str) -> "FusionMLPE2E":
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(**cfg)

    @classmethod
    def load(cls, config_path: str, checkpoint_path: str, map_location=None) -> "FusionMLPE2E":
        model = cls.from_config(config_path)
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model
