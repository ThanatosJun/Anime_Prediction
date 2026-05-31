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

import copy
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
      4. Normalize gates by sum of valid gates → consistent scale regardless of missing modalities
      5. Weighted average → [batch, 128]
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

        # normalize by sum of valid gates → consistent output scale
        gate_sum = gate.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [batch, 1, 1]
        out = (proj * (gate / gate_sum)).sum(dim=1)     # [batch, 128]
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

        # ── Modality flags（multimodal 消融用；未指定時全開 = 原行為）────────
        mods = config.get("modalities", {}) or {}
        self.use_image = mods.get("image", True)
        self.use_text  = mods.get("text",  True)
        self.use_meta  = mods.get("meta",  True)
        if self.use_rag and not self.use_meta:
            raise ValueError("use_rag=True 需要 meta 分支（meta 為 Cross Attention 的 Query）。")
        if not (self.use_image or self.use_text or self.use_meta):
            raise ValueError("image / text / meta 至少需啟用一個分支。")

        # ── Projection blocks（只建啟用的分支）──────────────────────────────
        if self.use_image:
            self.image_proj = ImageProjection(image_dim, proj_dim, proj_drop)
        if self.use_text:
            self.text_proj  = ProjectionBlock(text_dim,  proj_dim, proj_drop)
        if self.use_meta:
            self.meta_proj  = ProjectionBlock(meta_dim,  proj_dim, proj_drop)

        n_branches = int(self.use_image) + int(self.use_text) + int(self.use_meta)

        # ── Cross Attention（optional）────────────────────────────────────
        if self.use_rag:
            self.cross_attn = RAGCrossAttention(
                proj_dim    = proj_dim,
                num_heads   = cfg_m.get("attn_heads", 4),
                dropout     = proj_drop,
                attn_dropout= cfg_m.get("attn_dropout", 0.1),
            )
            concat_dim = proj_dim * (n_branches + 1)   # branches + cross_attn
        else:
            concat_dim = proj_dim * n_branches

        # ── MLP backbone ──────────────────────────────────────────────────
        self.backbone = MLPBackbone(concat_dim, hidden_dims, dropout)

        # ── Temporal Trend Head（optional）────────────────────────────────
        # backbone 輸出 Δ（殘差），trend_head 輸出年度基準
        # Final = Δ + trend，讓模型學習時序漂移
        self.use_trend_head = cfg_m.get("trend_head", {}).get("_active", False)
        if self.use_trend_head:
            self.trend_head = nn.Linear(1, 1)   # release_year（normalized）→ trend

    def forward(self, batch: dict, return_attn: bool = False):
        feats = []
        meta_feat = None
        if self.use_image:
            feats.append(self.image_proj(batch["image_emb"], batch["image_mask"]))
        if self.use_text:
            feats.append(self.text_proj(batch["text_emb"]))
        if self.use_meta:
            meta_feat = self.meta_proj(batch["meta_feat"])
            feats.append(meta_feat)

        attn_weights = None
        if self.use_rag:
            rag_out = self.cross_attn(
                query       = meta_feat,
                rag_meta    = batch["rag_meta"],
                rag_text    = batch["rag_text"],
                rag_image   = batch["rag_image"],
                rag_mask    = batch.get("rag_mask"),
                return_attn = return_attn,
            )
            cross_feat, attn_weights = (rag_out if return_attn
                                        else (rag_out, None))
            feats.append(cross_feat)

        fused = torch.cat(feats, dim=1)

        delta = self.backbone(fused)   # 殘差（anime-specific 偏差）

        if self.use_trend_head:
            # meta_feat[:, 0] = release_year（MetaEncoder 第一個特徵，已標準化）
            year  = batch["meta_feat"][:, 0:1]              # [batch, 1]
            trend = self.trend_head(year).squeeze(-1)        # [batch]
            out   = delta + trend
        else:
            out = delta

        return (out, attn_weights) if return_attn else out


# ── Config helper ─────────────────────────────────────────────────────────────

def make_model_config(config: dict, target: str) -> dict:
    """
    Return a per-target deep copy of config with trend_head._active correctly set.
    Use this everywhere a FusionModel is constructed to ensure consistent architecture.
    """
    mc = copy.deepcopy(config)
    th = mc["model"].get("trend_head", {})
    th["_active"] = th.get("enabled", False) and target in th.get("apply_to", [])
    mc["model"]["trend_head"] = th
    return mc
