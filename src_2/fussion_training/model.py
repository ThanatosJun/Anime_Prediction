"""
FusionModel v2

架構（proj_dim=128）：
  Image  [batch, 3, 1024] → ImageProjection (Shared + Gate) → [batch, 128]
  Text   [batch, 768]     → ProjectionBlock(768→128)        → [batch, 128]
  Meta   [batch, 56]      → ProjectionBlock(56→128)         → [batch, 128]
  RAG    Cross Attention                                     → [batch, 128]  ← use_rag=True 時
  ──────────────────────────────────────────────────────────────────────────
  concat: use_rag=True  → [batch, 512]  (128 × 4)
          use_rag=False → [batch, 384]  (128 × 3)
  ──────────────────────────────────────────────────────────────────────────
  MLP backbone: concat_dim → 256 → 128 → 1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from cross_attention import RAGCrossAttention


# ── Projection Block ──────────────────────────────────────────────────────────

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


# ── Image Projection with Shared Proj + Gating ───────────────────────────────

class ImageProjection(nn.Module):
    """
    輸入：image_emb [batch, 3, 1024]，image_mask [batch, 3] (True=缺失)
    輸出：[batch, 128]

    流程：
      1. Shared Linear(1024→128) + LN + GELU + Dropout → [batch, 3, 128]
      2. Gate = sigmoid(Linear(1024→1)) per modality   → [batch, 3, 1]
      3. gate=0 when mask=True（缺失模態不貢獻）
      4. Weighted sum → [batch, 128]
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = ProjectionBlock(in_dim, out_dim, dropout)
        self.gate = nn.Linear(in_dim, 1)   # content-based gate

    def forward(
        self,
        image_emb: torch.Tensor,    # [batch, 3, 1024]
        image_mask: torch.Tensor,   # [batch, 3] bool，True = 缺失
    ) -> torch.Tensor:              # [batch, 128]

        # projection
        proj = self.proj(image_emb)                     # [batch, 3, 128]

        # gate（從原始 embedding 計算）
        gate = torch.sigmoid(self.gate(image_emb))      # [batch, 3, 1]

        # 缺失模態的 gate 強制為 0
        gate = gate.masked_fill(image_mask.unsqueeze(-1), 0.0)

        # weighted sum
        out = (proj * gate).sum(dim=1)                  # [batch, 128]
        return out


# ── MLP Backbone ──────────────────────────────────────────────────────────────

class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [batch]


# ── Fusion Model ──────────────────────────────────────────────────────────────

class FusionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg_m = config["model"]

        proj_dim    = cfg_m.get("proj_dim", 128)
        dropout     = cfg_m.get("dropout", 0.3)
        proj_drop   = cfg_m.get("attn_dropout", 0.1)
        hidden_dims = cfg_m.get("hidden_dims", [256, 128])
        image_dim   = cfg_m.get("image_dim", 1024)
        text_dim    = cfg_m.get("text_dim",  768)
        meta_dim    = cfg_m.get("meta_dim",  56)
        self.use_rag = config.get("use_rag", True)

        # ── Projection blocks ──────────────────────────────────────────────
        self.image_proj = ImageProjection(image_dim, proj_dim, proj_drop)
        self.text_proj  = ProjectionBlock(text_dim,  proj_dim, proj_drop)
        self.meta_proj  = ProjectionBlock(meta_dim,  proj_dim, proj_drop)

        # ── Cross Attention（optional）────────────────────────────────────
        if self.use_rag:
            self.cross_attn = RAGCrossAttention(
                proj_dim    = proj_dim,
                num_heads   = cfg_m.get("attn_heads", 4),
                dropout     = proj_drop,
                attn_dropout= cfg_m.get("attn_dropout", 0.1),
            )
            concat_dim = proj_dim * 4   # image + text + meta + cross_attn
        else:
            concat_dim = proj_dim * 3   # image + text + meta

        # ── MLP backbone ──────────────────────────────────────────────────
        self.backbone = MLPBackbone(concat_dim, hidden_dims, dropout)

    def forward(self, batch: dict, return_attn: bool = False):
        image_feat = self.image_proj(batch["image_emb"], batch["image_mask"])
        text_feat  = self.text_proj(batch["text_emb"])
        meta_feat  = self.meta_proj(batch["meta_feat"])

        if self.use_rag:
            rag_mask = batch.get("rag_mask")
            rag_out = self.cross_attn(
                query       = meta_feat,
                rag_meta    = batch["rag_meta"],
                rag_text    = batch["rag_text"],
                rag_image   = batch["rag_image"],
                rag_mask    = rag_mask,
                return_attn = return_attn,
            )
            cross_feat, attn_weights = (rag_out if return_attn
                                        else (rag_out, None))
            fused = torch.cat([image_feat, text_feat, meta_feat, cross_feat], dim=1)
        else:
            fused = torch.cat([image_feat, text_feat, meta_feat], dim=1)
            attn_weights = None

        out = self.backbone(fused)
        return (out, attn_weights) if return_attn else out
