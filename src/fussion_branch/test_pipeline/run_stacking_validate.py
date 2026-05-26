"""
Stacking validation: checks whether LGBM(meta) and MLP(text+image) predictions
are complementary enough to benefit from ensembling.

Steps:
  1. Load FusionDataset for all splits (features already concatenated: text+image+meta)
  2. Train LGBM on meta_65 only
  3. Train tiny MLP on text+image (1408-dim) only
  4. Get val predictions from both
  5. Compute Spearman correlation between them
  6. Scan alpha in [0,1] → best log_MAE of alpha*ŷ_lgbm + (1-alpha)*ŷ_dl
  7. Print verdict

Usage:
    python -m src.fussion_branch.run_stacking_validate
    python -m src.fussion_branch.run_stacking_validate --target meanScore
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
import yaml

from src.fussion_branch.fussion_training.dataset import FusionDataset
from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder
from src.fussion_branch.utilities.evaluate import compute_metrics, denormalize
from src.fussion_branch.utilities.config import load_config


# ── tiny MLP for text+image only ──────────────────────────────────────────────

class TextImageMLP(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.LayerNorm(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_target_scaler(meta_df, target_col, log_transform, winsor_pct=None):
    y = meta_df[target_col].dropna().values.astype(np.float64)
    if log_transform:
        y = np.log1p(y)
    winsor_cap = None
    if winsor_pct is not None:
        winsor_cap = float(np.percentile(y, winsor_pct))
        y = np.clip(y, None, winsor_cap)
    return {"mean": float(y.mean()), "std": max(float(y.std()), 1e-8),
            "log_transform": log_transform, "winsor_cap": winsor_cap}


def _to_arrays(dataset):
    """Return (X_text_image, X_meta, y_norm) numpy arrays from FusionDataset."""
    all_feat, all_y = [], []
    for i in range(len(dataset)):
        s = dataset[i]
        all_feat.append(s["features"].numpy())
        all_y.append(s["target"].item())
    X = np.stack(all_feat)              # (N, text+image+meta)
    y = np.array(all_y, dtype=np.float32)

    t_dim = dataset.text_dim
    i_dim = dataset.image_dim
    X_emb  = X[:, : t_dim + i_dim]     # text + image
    X_meta = X[:, t_dim + i_dim :]     # meta only
    return X_emb, X_meta, y


def _train_lgbm(X_tr, y_tr, X_val, y_val):
    params = {
        "objective":       "regression",
        "metric":          "mae",
        "num_leaves":      63,
        "learning_rate":   0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":    5,
        "min_child_samples": 20,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "n_jobs":          -1,
        "verbose":         -1,
        "seed":            42,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    cb = lgb.early_stopping(50, verbose=False)
    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[cb],
    )
    return model


def _train_dl(X_tr, y_tr, X_val, y_val, device, epochs=100, patience=20):
    in_dim = X_tr.shape[1]
    model  = TextImageMLP(in_dim).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=5)
    crit   = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_val, best_state, cnt = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        # mini-batch SGD
        idx = torch.randperm(len(Xt))
        for start in range(0, len(Xt), 256):
            b = idx[start:start+256]
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = crit(model(Xt[b]), yt[b])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = crit(model(Xt[b]), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        with torch.no_grad():
            val_mae = float(torch.mean(torch.abs(model(Xv) - yv)).item())
        sched.step(val_mae)

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

        if ep % 20 == 0 or ep == 1:
            print(f"    [DL ep {ep:3d}]  val_MAE={val_mae:.4f}")

    model.load_state_dict(best_state)
    return model


def _predict_dl(model, X, device):
    model.eval()
    with torch.no_grad():
        preds = []
        for start in range(0, len(X), 512):
            batch = torch.from_numpy(X[start:start+512]).to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


# ── main ──────────────────────────────────────────────────────────────────────

def validate(target_col: str, config_path: str):
    cfg = load_config(config_path)
    cfg_data  = cfg["data"]
    log_transform = cfg["targets"][target_col]["log_transform"]
    winsor_pct    = cfg["targets"][target_col].get("winsor_pct", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Stacking Validation: {target_col}")
    print(f"  device={device}")
    print(f"{'='*60}")

    # ── MetaEncoder ──────────────────────────────────────────────────────────
    encoder_path = cfg_data["meta_encoder_path"]
    if Path(encoder_path).exists():
        encoder = MetaEncoder.load(encoder_path)
    else:
        meta_tr = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")
        rag_tr  = pd.read_parquet(f"{cfg_data['rag_features_dir']}/rag_features_train.parquet")
        encoder = MetaEncoder().fit(meta_tr, rag_tr)
        encoder.save(encoder_path)

    # ── target scaler ────────────────────────────────────────────────────────
    meta_tr = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")
    scaler  = _build_target_scaler(meta_tr, target_col, log_transform, winsor_pct)

    image_emb_dir = cfg_data.get("image_emb_dir", "src/fussion_branch/embedding/image")

    def make_ds(split, winsor=False):
        return FusionDataset(
            split=split, encoder=encoder,
            meta_dir=cfg_data["fusion_meta_dir"],
            text_emb_dir=cfg_data["text_emb_dir"],
            rag_dir=cfg_data["rag_features_dir"],
            image_emb_dir=image_emb_dir,
            target_col=target_col,
            log_transform_target=log_transform,
            target_mean=scaler["mean"], target_std=scaler["std"],
            winsor_cap=scaler.get("winsor_cap") if winsor else None,
        )

    print("\n[1] Loading datasets…")
    train_ds = make_ds("train", winsor=True)
    val_ds   = make_ds("val",   winsor=False)

    X_emb_tr,  X_meta_tr,  y_tr  = _to_arrays(train_ds)
    X_emb_val, X_meta_val, y_val = _to_arrays(val_ds)

    print(f"    train: {len(y_tr):,}  val: {len(y_val):,}")
    print(f"    text+image dim: {X_emb_tr.shape[1]}   meta dim: {X_meta_tr.shape[1]}")

    # ── LGBM on meta ─────────────────────────────────────────────────────────
    print("\n[2] Training LGBM on meta_65…")
    lgbm_model = _train_lgbm(X_meta_tr, y_tr, X_meta_val, y_val)
    ŷ_lgbm_val_norm = lgbm_model.predict(X_meta_val).astype(np.float32)

    y_val_orig   = denormalize(y_val,            scaler)
    ŷ_lgbm_orig  = denormalize(ŷ_lgbm_val_norm,  scaler)
    m_lgbm = compute_metrics(y_val_orig, ŷ_lgbm_orig, target_col)
    print(f"    LGBM(meta) val →  " + "  ".join(f"{k}={v}" for k, v in m_lgbm.items()))

    # ── DL on text+image only ─────────────────────────────────────────────────
    print("\n[3] Training MLP on text+image only…")
    dl_model = _train_dl(X_emb_tr, y_tr, X_emb_val, y_val, device)
    ŷ_dl_val_norm = _predict_dl(dl_model, X_emb_val, device)

    ŷ_dl_orig = denormalize(ŷ_dl_val_norm, scaler)
    m_dl = compute_metrics(y_val_orig, ŷ_dl_orig, target_col)
    print(f"    DL(text+img) val → " + "  ".join(f"{k}={v}" for k, v in m_dl.items()))

    # ── complementarity check ─────────────────────────────────────────────────
    print("\n[4] Complementarity analysis…")
    rho_preds, _ = spearmanr(ŷ_lgbm_val_norm, ŷ_dl_val_norm)
    rho_err_lgbm = float(spearmanr(ŷ_lgbm_orig - y_val_orig,
                                    ŷ_dl_orig  - y_val_orig)[0])
    print(f"    Spearman(ŷ_lgbm, ŷ_dl)          = {rho_preds:.4f}")
    print(f"    Spearman(err_lgbm, err_dl)       = {rho_err_lgbm:.4f}")

    if abs(rho_preds) > 0.95:
        verdict = "REDUNDANT — two models predict nearly the same thing; stacking unlikely to help"
    elif abs(rho_preds) > 0.85:
        verdict = "PARTIALLY COMPLEMENTARY — stacking may give small gains"
    else:
        verdict = "COMPLEMENTARY — stacking should help meaningfully"
    print(f"    Verdict: {verdict}")

    # ── alpha sweep ───────────────────────────────────────────────────────────
    print("\n[5] Scanning alpha (final = α·ŷ_lgbm + (1-α)·ŷ_dl)…")
    best_alpha, best_metric = 0.0, float("inf")
    metric_key = "log_MAE" if target_col == "popularity" else "MAE"

    results = []
    for alpha in np.linspace(0, 1, 21):
        blend_norm = alpha * ŷ_lgbm_val_norm + (1 - alpha) * ŷ_dl_val_norm
        blend_orig = denormalize(blend_norm, scaler)
        m = compute_metrics(y_val_orig, blend_orig, target_col)
        val = m[metric_key]
        results.append((round(alpha, 2), val, m.get("Spearman_rho")))
        if val < best_metric:
            best_metric = val
            best_alpha  = round(alpha, 2)

    print(f"    {'alpha':>6}  {metric_key:>10}  {'Spearman':>10}")
    for a, v, s in results:
        marker = " ←" if a == best_alpha else ""
        print(f"    {a:>6.2f}  {v:>10.4f}  {s:>10.4f}{marker}")

    print(f"\n  Best alpha={best_alpha}  best val {metric_key}={best_metric:.4f}")
    print(f"  LGBM alone:          val {metric_key}={m_lgbm[metric_key]:.4f}")
    print(f"  DL(text+img) alone:  val {metric_key}={m_dl[metric_key]:.4f}")

    gain = m_lgbm[metric_key] - best_metric
    print(f"  Gain from stacking:  Δ{metric_key}={gain:+.4f}")

    print("\n  Summary:")
    print(f"    Spearman(ŷ_lgbm, ŷ_dl) = {rho_preds:.4f}")
    print(f"    {'Stacking helps' if gain > 0.005 else 'Stacking does NOT help significantly'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="popularity",
                        choices=["popularity", "meanScore"])
    parser.add_argument("--config",
                        default="src/fussion_branch/configs/fusion_config.yaml")
    args = parser.parse_args()
    validate(args.target, args.config)


if __name__ == "__main__":
    main()
