"""
SHAP analysis for FusionMLP.

Usage:
    python -m src.fussion_branch.run_shap --target popularity
    python -m src.fussion_branch.run_shap --target meanScore
    python -m src.fussion_branch.run_shap --target popularity --n_background 200 --n_explain 500

Output (saved to results_dir/{run_id}/{target}/shap/):
    modality_importance.json   — mean |SHAP| per modality
    meta_bar.png               — top-N meta feature bar chart
    meta_beeswarm.png          — beeswarm for meta features
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

try:
    import shap
except ImportError:
    raise ImportError("pip install shap")

from src.fussion_branch.fussion_training.dataset import FusionDataset
from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder
from src.fussion_branch.fussion_training.model import FusionMLP


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: str = "src/fussion_branch/configs/fusion_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_features(loader, device, n_max: int) -> torch.Tensor:
    parts = []
    total = 0
    for batch in loader:
        parts.append(batch["features"])
        total += len(batch["features"])
        if total >= n_max:
            break
    x = torch.cat(parts, dim=0)[:n_max]
    return x.to(device)


# ── device-aware wrappers ─────────────────────────────────────────────────────
# SHAP's GradientExplainer operates on CPU tensors internally; these wrappers
# move input to the model's device on each forward call so SHAP never needs to
# know about CUDA.

class _FullWrapper(torch.nn.Module):
    def __init__(self, model: FusionMLP, device: torch.device):
        super().__init__()
        self.model   = model
        self._device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self._device)).unsqueeze(-1)  # (batch,) → (batch, 1)


class _MetaWrapper(torch.nn.Module):
    """Fix text+image projections; only accept meta slice as input.
    Lets SHAP attribute contributions to the 65 meta features directly.
    """
    def __init__(self, model: FusionMLP, fixed_t: torch.Tensor, fixed_img: torch.Tensor,
                 device: torch.device):
        super().__init__()
        self.model   = model
        self._device = device
        # store on CPU; moved to device in forward
        self._fixed_t   = fixed_t.cpu()
        self._fixed_img = fixed_img.cpu()

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        dev = self._device
        B   = meta.shape[0]
        t   = self._fixed_t.to(dev).expand(B, -1)
        img = self._fixed_img.to(dev).expand(B, -1)
        m   = self.model.meta_proj(meta.to(dev))
        fused = torch.cat([t, img, m], dim=-1)
        return self.model.head(self.model.backbone(fused))  # (batch, 1) — no squeeze for SHAP


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",       default="popularity", choices=["popularity", "meanScore"])
    parser.add_argument("--n_background", type=int, default=200,  help="SHAP background samples")
    parser.add_argument("--n_explain",    type=int, default=500,  help="samples to explain")
    parser.add_argument("--top_n",        type=int, default=20,   help="top N meta features to plot")
    parser.add_argument("--config",       default="src/fussion_branch/configs/fusion_config.yaml")
    args = parser.parse_args()

    config   = _load_config(args.config)
    cfg_data = config["data"]
    cfg_out  = config["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    run_dir = Path(cfg_out["results_dir"]) / cfg_out["run_id"] / args.target
    model = FusionMLP.load(
        str(run_dir / "model_config.json"),
        str(run_dir / "best_model.pt"),
        map_location=device,
    ).to(device).eval()
    print(f"model loaded from {run_dir}")

    # ── load encoder + dataset ────────────────────────────────────────────────
    encoder = MetaEncoder.load(cfg_data["meta_encoder_path"])
    feature_names = encoder.feature_names_
    assert len(feature_names) == encoder.feature_dim, \
        f"feature_names length {len(feature_names)} ≠ feature_dim {encoder.feature_dim}"
    print(f"meta feature_dim={encoder.feature_dim}, names OK")

    from torch.utils.data import DataLoader
    import pandas as pd

    def make_ds(split):
        return FusionDataset(
            split=split,
            encoder=encoder,
            meta_dir=cfg_data["fusion_meta_dir"],
            text_emb_dir=cfg_data["text_emb_dir"],
            rag_dir=cfg_data["rag_features_dir"],
            image_emb_dir=cfg_data.get("image_emb_dir", "src/fussion_branch/embedding/image"),
            target_col=args.target,
            log_transform_target=config["targets"][args.target]["log_transform"],
            target_mean=0.0,
            target_std=1.0,
        )

    val_ds = make_ds("val")
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    # ── collect full feature tensors ──────────────────────────────────────────
    n_total = args.n_background + args.n_explain
    all_x = _collect_features(val_loader, device, n_total)

    # text / image / meta boundaries
    text_dim  = model._text_dim
    image_dim = model._image_dim
    s1 = text_dim
    s2 = text_dim + image_dim

    background_x = all_x[:args.n_background]
    explain_x    = all_x[args.n_background:args.n_background + args.n_explain]
    print(f"background: {background_x.shape}  explain: {explain_x.shape}")

    # ── 1. full-model SHAP → modality-level importance ────────────────────────
    # Pass CPU tensors to SHAP; _FullWrapper moves them to GPU in forward.
    print("\n[1/2] full-model GradientExplainer …")
    full_wrapper   = _FullWrapper(model, device).eval()
    explainer_full = shap.GradientExplainer(full_wrapper, background_x.cpu())
    sv_full   = explainer_full.shap_values(explain_x.cpu())
    shap_full = np.array(sv_full[0] if isinstance(sv_full, list) else sv_full)  # (n, 1473)

    def _modality_stats(arr: np.ndarray) -> dict:
        """arr: (n_samples, n_dims) — return sum/mean/dim stats."""
        abs_arr = np.abs(arr)
        return {
            "dims":         arr.shape[1],
            "sum_abs_shap": float(abs_arr.sum(axis=1).mean()),   # sum over dims, mean over samples
            "mean_abs_shap": float(abs_arr.mean()),               # mean over dims AND samples
        }

    cfg_model = config["model"]
    proj_dims = {
        "text":  cfg_model["text_proj"],   # e.g. 128
        "image": cfg_model["image_proj"],  # e.g. 256
        "meta":  cfg_model["meta_proj"],   # e.g. 64
    }

    text_stats  = _modality_stats(shap_full[:, :s1])
    image_stats = _modality_stats(shap_full[:, s1:s2])
    meta_stats  = _modality_stats(shap_full[:, s2:])

    total_sum = text_stats["sum_abs_shap"] + image_stats["sum_abs_shap"] + meta_stats["sum_abs_shap"]

    modality_result = {}
    for mod, stats in [("text", text_stats), ("image", image_stats), ("meta", meta_stats)]:
        modality_result[mod] = {
            "input_dims":     stats["dims"],          # SHAP 歸因的維度（projection 前）
            "proj_dims":      proj_dims[mod],         # projection 後進 backbone 的維度
            "sum_abs_shap":   round(stats["sum_abs_shap"],  6),
            "mean_abs_shap":  round(stats["mean_abs_shap"], 6),
            "pct_by_sum":     round(stats["sum_abs_shap"] / total_sum * 100, 2),
        }

    print("\nModality importance:")
    print(f"  {'':6s}  {'in_dim':>6}  {'proj':>5}  {'sum':>10}  {'mean':>10}  {'pct(sum)':>10}")
    for mod, vals in modality_result.items():
        print(f"  {mod:6s}  {vals['input_dims']:>6}  {vals['proj_dims']:>5}  "
              f"{vals['sum_abs_shap']:>10.5f}  {vals['mean_abs_shap']:>10.5f}  "
              f"{vals['pct_by_sum']:>9.1f}%")

    # ── 2. meta-wrapper SHAP → named meta feature importance ─────────────────
    print("\n[2/2] meta-wrapper GradientExplainer …")
    with torch.no_grad():
        fixed_t   = model.text_proj(background_x[:, :s1]).mean(dim=0, keepdim=True)
        fixed_img = model.image_proj(background_x[:, s1:s2]).mean(dim=0, keepdim=True)

    wrapper = _MetaWrapper(model, fixed_t, fixed_img, device).eval()

    bg_meta  = background_x[:, s2:].cpu()
    exp_meta = explain_x[:, s2:].cpu()

    explainer_meta = shap.GradientExplainer(wrapper, bg_meta)
    sv_meta   = explainer_meta.shap_values(exp_meta)
    shap_meta = np.array(sv_meta[0] if isinstance(sv_meta, list) else sv_meta)
    if shap_meta.ndim == 3:
        shap_meta = shap_meta.squeeze(-1)   # (n, 65, 1) → (n, 65)
    print(f"  shap_meta shape: {shap_meta.shape}")

    # ── save outputs ──────────────────────────────────────────────────────────
    out_dir = run_dir / "shap"
    out_dir.mkdir(exist_ok=True)

    # modality json
    with open(out_dir / "modality_importance.json", "w") as f:
        json.dump(modality_result, f, indent=2)

    # meta bar chart
    mean_abs = np.abs(shap_meta).mean(axis=0)          # (65,)
    top_idx  = np.argsort(mean_abs)[::-1][:args.top_n]
    top_vals = mean_abs[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_names)), top_vals[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Meta feature importance — {args.target} (top {args.top_n})")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(str(out_dir / "meta_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  meta_bar.png saved")

    # beeswarm via shap library (creates its own figure internally)
    expl_obj = shap.Explanation(
        values=shap_meta,
        data=exp_meta.cpu().numpy(),
        feature_names=feature_names,
    )
    shap.plots.beeswarm(expl_obj, max_display=args.top_n, show=False)
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig(str(out_dir / "meta_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  meta_beeswarm.png saved")

    print(f"\nDone. Results → {out_dir}/")


if __name__ == "__main__":
    main()
