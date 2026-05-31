"""
Fusion Model 特徵重要性分析

1. Captum IntegratedGradients：各 modality 層級貢獻
   - image（yolo / cover / banner）、text、meta、rag
2. SHAP DeepExplainer：meta features 各維度貢獻（56 可解釋維度）

前置安裝：
    pip install captum shap

用法：
    python src_2/explain/feature_attr.py --target popularity --split val --n 20
    python src_2/explain/feature_attr.py --target popularity --ids 12345 67890
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fussion_training"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from dataset import AnimeDataset, denormalize_target
from meta_encoder import MetaEncoder
from model import FusionModel


# ── Captum wrapper ────────────────────────────────────────────────────────────

class FusionWrapper(torch.nn.Module):
    """把 FusionModel dict 介面包成 Captum 需要的多 tensor 輸入介面。"""

    def __init__(self, model: FusionModel, fixed_batch: dict):
        super().__init__()
        self.model       = model
        self.fixed_batch = fixed_batch   # 非目標 input 固定於此

    def forward(self, image_emb, text_emb, meta_feat,
                rag_meta, rag_text, rag_image):
        batch = dict(self.fixed_batch)
        batch["image_emb"]  = image_emb
        batch["text_emb"]   = text_emb
        batch["meta_feat"]  = meta_feat
        if self.model.use_rag:
            batch["rag_meta"]  = rag_meta
            batch["rag_text"]  = rag_text
            batch["rag_image"] = rag_image
        return self.model(batch)


def captum_modality_importance(
    model: FusionModel,
    ds: AnimeDataset,
    indices: list[int],
    device: torch.device,
    n_steps: int = 50,
) -> pd.DataFrame:
    """
    用 IntegratedGradients 計算每筆 sample 的各 modality 重要性。

    回傳 DataFrame，欄位：id, image_yolo, image_cover, image_banner,
                             text, meta, rag_meta, rag_text, rag_image
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise ImportError("pip install captum")

    rows = []
    for idx in indices:
        item  = ds[idx]
        aid   = ds.ids[idx]

        # baseline = zeros
        def _t(key):
            return item[key].unsqueeze(0).to(device)

        fixed = {
            "image_mask": item["image_mask"].unsqueeze(0).to(device),
            "rag_mask":   item["rag_mask"].unsqueeze(0).to(device) if "rag_mask" in item else None,
            "id": aid,
        }

        wrapper = FusionWrapper(model, fixed)
        ig      = IntegratedGradients(wrapper)

        inputs = (
            _t("image_emb"),   # [1, 3, 1024]
            _t("text_emb"),    # [1, 768]
            _t("meta_feat"),   # [1, 56]
            _t("rag_meta")  if "rag_meta"  in item else torch.zeros(1, 5, 10,   device=device),
            _t("rag_text")  if "rag_text"  in item else torch.zeros(1, 5, 768,  device=device),
            _t("rag_image") if "rag_image" in item else torch.zeros(1, 5, 1024, device=device),
        )
        baselines = tuple(torch.zeros_like(x) for x in inputs)

        attrs = ig.attribute(inputs, baselines=baselines,
                             n_steps=n_steps, return_convergence_delta=False)

        # aggregate: sum |attr| per modality
        img_attr = attrs[0].abs().squeeze(0)        # [3, 1024]
        row = {
            "id":           aid,
            "image_yolo":   float(img_attr[0].sum()),
            "image_cover":  float(img_attr[1].sum()),
            "image_banner": float(img_attr[2].sum()),
            "text":         float(attrs[1].abs().sum()),
            "meta":         float(attrs[2].abs().sum()),
            "rag_meta":     float(attrs[3].abs().sum()),
            "rag_text":     float(attrs[4].abs().sum()),
            "rag_image":    float(attrs[5].abs().sum()),
        }
        # normalize to sum = 1
        total = sum(v for k, v in row.items() if k != "id") + 1e-12
        for k in list(row.keys()):
            if k != "id":
                row[k] = round(row[k] / total, 4)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_modality_importance(df: pd.DataFrame, save_path: Path):
    modality_cols = [c for c in df.columns if c != "id"]
    means = df[modality_cols].mean()

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = plt.cm.Set2(np.linspace(0, 1, len(means)))
    ax.barh(means.index, means.values, color=colors)
    ax.set_xlabel("Mean |IG| (normalized)")
    ax.set_title("Modality Importance (Captum Integrated Gradients)")
    for i, v in enumerate(means.values):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── SHAP on meta features ─────────────────────────────────────────────────────

class MetaOnlyWrapper(torch.nn.Module):
    """固定其他 input，只讓 meta_feat 可變，供 SHAP 使用。"""

    def __init__(self, model: FusionModel, fixed_batch: dict):
        super().__init__()
        self.model       = model
        self.fixed_batch = fixed_batch

    def forward(self, meta_feat: torch.Tensor) -> torch.Tensor:
        batch = dict(self.fixed_batch)
        batch["meta_feat"] = meta_feat
        return self.model(batch).unsqueeze(-1)   # SHAP 需要 2-D 輸出


def shap_meta_importance(
    model: FusionModel,
    ds: AnimeDataset,
    indices: list[int],
    background_n: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """
    回傳：
      shap_values : [n_samples, 56]
      feature_names: MetaEncoder 的 56 個特徵名稱
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    model.eval()
    meta_encoder = ds.__class__   # get MetaEncoder from ds
    feature_names = (getattr(ds, "_meta_encoder_ref", None)
                     or [f"meta_{i}" for i in range(56)])

    # background：從 dataset 隨機取樣
    bg_idx = np.random.choice(len(ds), min(background_n, len(ds)), replace=False)
    bg_meta = torch.stack([ds[i]["meta_feat"] for i in bg_idx]).to(device)

    # 固定其他 input（用第一筆 sample 的值作為代表）
    sample0 = ds[indices[0]]
    fixed = {k: sample0[k].unsqueeze(0).to(device)
             for k in ("image_emb", "image_mask", "text_emb",
                       "rag_meta", "rag_text", "rag_image", "rag_mask")
             if k in sample0}

    wrapper = MetaOnlyWrapper(model, fixed)
    explainer = shap.DeepExplainer(wrapper, bg_meta)

    test_meta = torch.stack([ds[i]["meta_feat"] for i in indices]).to(device)
    shap_vals  = explainer.shap_values(test_meta)   # [n, 56, 1] or [n, 56]
    shap_vals  = np.array(shap_vals).squeeze(-1) if np.array(shap_vals).ndim == 3 \
                 else np.array(shap_vals)           # [n, 56]

    return shap_vals, feature_names


def plot_shap_summary(shap_vals: np.ndarray, feature_names: list[str],
                      top_k: int, save_path: Path):
    mean_abs = np.abs(shap_vals).mean(axis=0)  # [56]
    top_idx  = np.argsort(mean_abs)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35)))
    colors = ["#e74c3c" if mean_abs[i] > np.median(mean_abs[top_idx]) else "#3498db"
              for i in top_idx]
    ax.barh([feature_names[i] for i in top_idx[::-1]],
            mean_abs[top_idx[::-1]], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_k} Meta Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="src_2/fussion_configs.yaml")
    parser.add_argument("--target",      required=True, choices=["popularity", "meanScore"])
    parser.add_argument("--split",       default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--ids",         nargs="+", type=int, default=None)
    parser.add_argument("--n",           type=int, default=20,
                        help="隨機抽樣數量（--ids 未指定時）")
    parser.add_argument("--background",  type=int, default=100,
                        help="SHAP background sample 數")
    parser.add_argument("--top_k",       type=int, default=20,
                        help="SHAP summary 顯示前 k 個 meta feature")
    parser.add_argument("--ig_steps",   type=int, default=50)
    parser.add_argument("--skip_captum", action="store_true")
    parser.add_argument("--skip_shap",   action="store_true")
    parser.add_argument("--out_dir",     default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg_out = config["output"]
    run_dir = Path(cfg_out["run_dir"]) / cfg_out["run_id"] / args.target
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "explain" / "feature"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_encoder = MetaEncoder.load(config["data"]["meta_encoder_path"])
    with open(run_dir / "target_scaler.json") as f:
        target_scaler = json.load(f)

    ds = AnimeDataset(args.split, config, meta_encoder,
                      target=args.target, target_scaler=target_scaler)

    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model.eval()

    # 選取 sample indices
    all_ids = ds.ids
    if args.ids:
        indices = [all_ids.index(i) for i in args.ids if i in all_ids]
    else:
        indices = list(np.random.choice(len(ds), min(args.n, len(ds)), replace=False))

    # ── Captum ────────────────────────────────────────────────────────────────
    if not args.skip_captum:
        print(f"Running Captum IG on {len(indices)} samples...")
        df_cap = captum_modality_importance(model, ds, indices, device, args.ig_steps)
        csv_path = out_dir / "captum_modality.csv"
        df_cap.to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")

        fig_path = out_dir / "captum_modality.png"
        plot_modality_importance(df_cap, fig_path)
        print(f"  Saved → {fig_path}")
        print(df_cap.drop(columns="id").mean().to_string())

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if not args.skip_shap:
        print(f"\nRunning SHAP on {len(indices)} samples (background={args.background})...")
        feature_names = getattr(meta_encoder, "feature_names_",
                                [f"meta_{i}" for i in range(56)])
        # attach to ds for access inside shap_meta_importance
        ds._meta_encoder_ref = feature_names

        shap_vals, feat_names = shap_meta_importance(
            model, ds, indices, args.background, device
        )
        np.save(out_dir / "shap_values.npy", shap_vals)

        fig_path = out_dir / "shap_summary.png"
        plot_shap_summary(shap_vals, feat_names, args.top_k, fig_path)
        print(f"  Saved → {fig_path}")

        # Top features table
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:args.top_k]
        for i in top_idx:
            print(f"  {feat_names[i]:35s}  {mean_abs[i]:.4f}")


if __name__ == "__main__":
    main()
