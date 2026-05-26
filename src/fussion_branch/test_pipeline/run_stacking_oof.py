"""
Full stacking pipeline with OOF (Out-of-Fold) predictions.

Level 0 base models (train on K-1 folds, predict fold-out):
  - LGBM(meta_65)
  - E2E text-only MLP (fine-tuned MiniLM-L6, no image)

Level 1 meta-model:
  - Ridge regression on [ŷ_lgbm_oof, ŷ_dl_oof]

Final inference:
  - Retrain both base models on full training set
  - Apply fitted Ridge → val / test predictions

Usage:
    python -m src.fussion_branch.run_stacking_oof
    python -m src.fussion_branch.run_stacking_oof --target meanScore --folds 5
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from fussion_branch.test_pipeline.dataset_e2e import FusionDatasetE2E
from fussion_branch.test_pipeline.model_e2e import FusionMLPE2E
from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder
from src.fussion_branch.utilities.evaluate import compute_metrics, denormalize
from src.fussion_branch.utilities.config import load_config


# ── constants ─────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "objective": "regression", "metric": "mae",
    "num_leaves": 63, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "n_jobs": -1, "verbose": -1, "seed": 42,
}

E2E_CFG = {
    "encoder_name":  "sentence-transformers/all-MiniLM-L6-v2",
    "freeze_layers": 4,
    "text_proj":     128,
    "image_proj":    64,   # unused (no_image=True)
    "meta_proj":     64,   # unused (no_meta=True)
    "hidden_dims":   [256, 128, 64],
    "dropout":       0.4,
    "no_image":      True,
    "no_meta":       True,   # pure text — more orthogonal to LGBM
}
E2E_TRAIN_CFG = {
    "backbone_lr":   1e-5,
    "head_lr":       5e-4,
    "weight_decay":  1e-2,
    "batch_size":    64,
    "epochs":        100,
    "warmup":        5,
    "patience":      20,
    "grad_clip":     1.0,
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_scaler(meta_df, target_col, log_transform, winsor_pct=None):
    y = meta_df[target_col].dropna().values.astype(np.float64)
    if log_transform:
        y = np.log1p(y)
    cap = None
    if winsor_pct is not None:
        cap = float(np.percentile(y, winsor_pct))
        y = np.clip(y, None, cap)
    return {"mean": float(y.mean()), "std": max(float(y.std()), 1e-8),
            "log_transform": log_transform, "winsor_cap": cap}


def _train_lgbm(X_tr, y_tr, X_val, y_val):
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    return lgb.train(LGBM_PARAMS, dtrain, num_boost_round=1000,
                     valid_sets=[dval],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def _train_e2e(train_subset, val_subset, meta_dim, device, use_amp):
    model = FusionMLPE2E(
        image_dim=1024, meta_dim=meta_dim, **E2E_CFG,
    ).to(device)

    cfg = E2E_TRAIN_CFG
    optimizer = torch.optim.AdamW(
        [{"params": model.backbone_parameters(), "lr": cfg["backbone_lr"]},
         {"params": model.head_parameters(),     "lr": cfg["head_lr"]}],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=5, min_lr=1e-7)
    criterion = nn.HuberLoss(delta=1.0)
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    tr_loader = DataLoader(train_subset, batch_size=cfg["batch_size"],
                           shuffle=True, num_workers=2, pin_memory=True)
    vl_loader = DataLoader(val_subset,   batch_size=cfg["batch_size"],
                           shuffle=False, num_workers=2, pin_memory=True)

    best_mae, best_state, patience_cnt = float("inf"), None, 0
    dummy_img = torch.zeros(1, 1024, device=device)

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch <= cfg["warmup"]:
            optimizer.param_groups[1]["lr"] = cfg["head_lr"] * epoch / cfg["warmup"]

        model.train()
        for batch in tr_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            meta = batch["meta_feat"].to(device)
            img  = dummy_img.expand(ids.shape[0], -1)
            y    = batch["target"].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(ids, mask, img, meta), y)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                amp_scaler.step(optimizer); amp_scaler.update()
            else:
                loss = criterion(model(ids, mask, img, meta), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

        model.eval()
        maes = []
        with torch.no_grad():
            for batch in vl_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                meta = batch["meta_feat"].to(device)
                img  = dummy_img.expand(ids.shape[0], -1)
                y    = batch["target"].to(device)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = model(ids, mask, img, meta)
                else:
                    pred = model(ids, mask, img, meta)
                maes.append(torch.mean(torch.abs(pred - y)).item() * len(y))
        val_mae = sum(maes) / len(val_subset)
        if epoch > cfg["warmup"]:
            scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model


def _predict_e2e(model, dataset, device, use_amp):
    dummy_img = torch.zeros(1, 1024, device=device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=128, shuffle=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            meta = batch["meta_feat"].to(device)
            img  = dummy_img.expand(ids.shape[0], -1)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    p = model(ids, mask, img, meta)
            else:
                p = model(ids, mask, img, meta)
            preds.append(p.cpu().numpy())
    return np.concatenate(preds)


# ── main pipeline ──────────────────────────────────────────────────────────────

def run_stacking(target_col: str, config_path: str, n_folds: int = 5):
    cfg      = load_config(config_path)
    cfg_data = cfg["data"]
    log_transform = cfg["targets"][target_col]["log_transform"]
    winsor_pct    = cfg["targets"][target_col].get("winsor_pct", None)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp  = device.type == "cuda"

    print(f"\n{'='*60}")
    print(f"  Stacking OOF: {target_col}  ({n_folds}-fold)")
    print(f"  device={device}")
    print(f"{'='*60}")

    # ── MetaEncoder & scaler ─────────────────────────────────────────────────
    encoder = MetaEncoder.load(cfg_data["meta_encoder_path"])
    meta_tr = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")
    scaler  = _build_scaler(meta_tr, target_col, log_transform, winsor_pct)

    # ── load full train + val as E2E dataset (text-only, no image) ───────────
    tokenizer = AutoTokenizer.from_pretrained(
        E2E_CFG["encoder_name"], clean_up_tokenization_spaces=True)

    def make_e2e(split, winsor=False):
        return FusionDatasetE2E(
            split=split, encoder=encoder, tokenizer=tokenizer, max_length=128,
            meta_dir=cfg_data["fusion_meta_dir"],
            rag_dir=cfg_data["rag_features_dir"],
            image_emb_dir=None,          # text-only
            target_col=target_col,
            log_transform_target=log_transform,
            target_mean=scaler["mean"], target_std=scaler["std"],
            winsor_cap=scaler["winsor_cap"] if winsor else None,
        )

    print("\n[0] Loading datasets…")
    full_train = make_e2e("train", winsor=True)
    val_ds     = make_e2e("val",   winsor=False)
    test_ds    = make_e2e("test",  winsor=False)
    meta_dim   = full_train.meta_dim
    N_train    = len(full_train)
    print(f"    train={N_train:,}  val={len(val_ds):,}  test={len(test_ds):,}  meta_dim={meta_dim}")

    # ── load pre-computed text embeddings for LGBM hybrid ────────────────────
    text_emb_dir = cfg_data.get("text_emb_dir", "src/fussion_branch/embedding/text")

    def _load_text_emb(split, ids: np.ndarray) -> np.ndarray:
        df = pd.read_parquet(f"{text_emb_dir}/text_embeddings_{split}.parquet")
        df = df.set_index("id")
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        # align to dataset IDs; fill missing with zeros
        aligned = df.reindex(ids)[emb_cols].fillna(0.0)
        return aligned.values.astype(np.float32)

    train_ids = full_train.ids
    val_ids   = val_ds.ids
    test_ids  = test_ds.ids

    print("    Loading text embeddings for LGBM hybrid…")
    text_emb_tr   = _load_text_emb("train", train_ids)
    text_emb_val  = _load_text_emb("val",   val_ids)
    text_emb_test = _load_text_emb("test",  test_ids)
    print(f"    text_emb_dim={text_emb_tr.shape[1]}  "
          f"LGBM input_dim={full_train.meta_feat.shape[1] + text_emb_tr.shape[1]}")

    # LGBM: hybrid = meta_65 + text_emb_384 = 449 dims
    X_lgbm_tr   = np.concatenate([full_train.meta_feat, text_emb_tr],   axis=1)
    X_lgbm_val  = np.concatenate([val_ds.meta_feat,    text_emb_val],   axis=1)
    X_lgbm_test = np.concatenate([test_ds.meta_feat,   text_emb_test],  axis=1)

    y_full = full_train.target   # (N_train,) — normalised

    # ── K-fold OOF ───────────────────────────────────────────────────────────
    kf = KFold(n_splits=n_folds, shuffle=False)   # no shuffle → preserve time order
    oof_lgbm = np.zeros(N_train, dtype=np.float32)
    oof_dl   = np.zeros(N_train, dtype=np.float32)

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(np.arange(N_train)), 1):
        print(f"\n── Fold {fold}/{n_folds}  "
              f"(train={len(tr_idx)}, oof_val={len(vl_idx)}) ──")

        # ── LGBM OOF (hybrid: meta + text_emb) ───────────────────────────────
        lgbm_model = _train_lgbm(
            X_lgbm_tr[tr_idx], y_full[tr_idx],
            X_lgbm_tr[vl_idx], y_full[vl_idx],
        )
        oof_lgbm[vl_idx] = lgbm_model.predict(X_lgbm_tr[vl_idx]).astype(np.float32)
        print(f"    LGBM OOF fold MAE = "
              f"{np.mean(np.abs(oof_lgbm[vl_idx] - y_full[vl_idx])):.4f}")

        # ── E2E OOF ───────────────────────────────────────────────────────────
        tr_sub = Subset(full_train, tr_idx.tolist())
        vl_sub = Subset(full_train, vl_idx.tolist())
        e2e_model = _train_e2e(tr_sub, vl_sub, meta_dim, device, use_amp)
        oof_dl[vl_idx] = _predict_e2e(e2e_model, vl_sub, device, use_amp)
        print(f"    E2E  OOF fold MAE = "
              f"{np.mean(np.abs(oof_dl[vl_idx] - y_full[vl_idx])):.4f}")
        del e2e_model

    # ── OOF metrics ───────────────────────────────────────────────────────────
    y_orig_tr = denormalize(y_full, scaler)
    m_oof_lgbm = compute_metrics(y_orig_tr, denormalize(oof_lgbm, scaler), target_col)
    m_oof_dl   = compute_metrics(y_orig_tr, denormalize(oof_dl,   scaler), target_col)
    metric_key = "log_MAE" if target_col == "popularity" else "MAE"
    print(f"\n[OOF summary]")
    print(f"  LGBM OOF {metric_key} = {m_oof_lgbm[metric_key]:.4f}")
    print(f"  E2E  OOF {metric_key} = {m_oof_dl[metric_key]:.4f}")

    # ── Ridge meta-model ─────────────────────────────────────────────────────
    print("\n[1] Training Ridge meta-model on OOF…")
    X_oof = np.column_stack([oof_lgbm, oof_dl])   # (N_train, 2)

    best_ridge, best_oof_score = None, float("inf")
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_oof, y_full)
        pred_oof = ridge.predict(X_oof)
        score = float(np.mean(np.abs(pred_oof - y_full)))
        if score < best_oof_score:
            best_oof_score = score
            best_ridge = ridge

    print(f"    Ridge coefs: lgbm={best_ridge.coef_[0]:.4f}  "
          f"dl={best_ridge.coef_[1]:.4f}  "
          f"intercept={best_ridge.intercept_:.4f}")

    # ── retrain base models on full training set ──────────────────────────────
    print("\n[2] Retraining base models on full training set…")

    lgbm_full = _train_lgbm(X_lgbm_tr, y_full, X_lgbm_val, val_ds.target)
    y_lgbm_val  = lgbm_full.predict(X_lgbm_val).astype(np.float32)
    y_lgbm_test = lgbm_full.predict(X_lgbm_test).astype(np.float32)
    print(f"    LGBM full val  {metric_key} = "
          f"{compute_metrics(denormalize(val_ds.target,scaler), denormalize(y_lgbm_val,scaler), target_col)[metric_key]:.4f}")
    print(f"    LGBM full test {metric_key} = "
          f"{compute_metrics(denormalize(test_ds.target,scaler), denormalize(y_lgbm_test,scaler), target_col)[metric_key]:.4f}")

    e2e_full  = _train_e2e(full_train, val_ds, meta_dim, device, use_amp)
    y_dl_val  = _predict_e2e(e2e_full, val_ds,  device, use_amp)
    y_dl_test = _predict_e2e(e2e_full, test_ds, device, use_amp)
    print(f"    E2E  full val  {metric_key} = "
          f"{compute_metrics(denormalize(val_ds.target,scaler), denormalize(y_dl_val,scaler), target_col)[metric_key]:.4f}")
    print(f"    E2E  full test {metric_key} = "
          f"{compute_metrics(denormalize(test_ds.target,scaler), denormalize(y_dl_test,scaler), target_col)[metric_key]:.4f}")

    # ── final stacked predictions (val + test) ────────────────────────────────
    print("\n[3] Applying Ridge meta-model to val and test…")
    y_stack_val  = best_ridge.predict(np.column_stack([y_lgbm_val,  y_dl_val ])).astype(np.float32)
    y_stack_test = best_ridge.predict(np.column_stack([y_lgbm_test, y_dl_test])).astype(np.float32)

    y_val_orig   = denormalize(val_ds.target,  scaler)
    y_test_orig  = denormalize(test_ds.target, scaler)

    m_lgbm_val  = compute_metrics(y_val_orig,  denormalize(y_lgbm_val,  scaler), target_col)
    m_lgbm_test = compute_metrics(y_test_orig, denormalize(y_lgbm_test, scaler), target_col)
    m_dl_val    = compute_metrics(y_val_orig,  denormalize(y_dl_val,    scaler), target_col)
    m_dl_test   = compute_metrics(y_test_orig, denormalize(y_dl_test,   scaler), target_col)
    m_stack_val  = compute_metrics(y_val_orig,  denormalize(y_stack_val,  scaler), target_col)
    m_stack_test = compute_metrics(y_test_orig, denormalize(y_stack_test, scaler), target_col)

    print(f"\n{'='*60}")
    print(f"  Final Results — {target_col}")
    print(f"{'='*60}")
    print(f"  {'Method':<26} {'val '+metric_key:>12}  {'test '+metric_key:>12}  {'val Spearman':>13}")
    print(f"  {'-'*66}")
    print(f"  {'LGBM(meta) alone':<26} {m_lgbm_val[metric_key]:>12.4f}  {m_lgbm_test[metric_key]:>12.4f}  {m_lgbm_val['Spearman_rho']:>13.4f}")
    print(f"  {'E2E text-only alone':<26} {m_dl_val[metric_key]:>12.4f}  {m_dl_test[metric_key]:>12.4f}  {m_dl_val['Spearman_rho']:>13.4f}")
    print(f"  {'Stacking (Ridge OOF)':<26} {m_stack_val[metric_key]:>12.4f}  {m_stack_test[metric_key]:>12.4f}  {m_stack_val['Spearman_rho']:>13.4f}")
    gain_val  = m_lgbm_val[metric_key]  - m_stack_val[metric_key]
    gain_test = m_lgbm_test[metric_key] - m_stack_test[metric_key]
    print(f"\n  Gain vs LGBM (val):  Δ{metric_key}={gain_val:+.4f}")
    print(f"  Gain vs LGBM (test): Δ{metric_key}={gain_test:+.4f}")
    shift = m_stack_test[metric_key] - m_stack_val[metric_key]
    print(f"  Val→Test shift:      Δ{metric_key}={shift:+.4f}  "
          f"({'expected temporal drift' if shift > 0 else 'better on test'})")

    # save results
    out_dir = Path(".exp/fussion/stacking") / target_col
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "target": target_col, "n_folds": n_folds,
        "ridge_coef_lgbm": float(best_ridge.coef_[0]),
        "ridge_coef_dl":   float(best_ridge.coef_[1]),
        "ridge_intercept": float(best_ridge.intercept_),
        "val":  {"lgbm": m_lgbm_val,  "e2e": m_dl_val,  "stack": m_stack_val},
        "test": {"lgbm": m_lgbm_test, "e2e": m_dl_test, "stack": m_stack_test},
        "oof":  {"lgbm": m_oof_lgbm,  "e2e": m_oof_dl},
    }
    with open(out_dir / "stacking_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_dir}/stacking_results.json")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="popularity",
                        choices=["popularity", "meanScore"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--config",
                        default="src/fussion_branch/configs/fusion_config.yaml")
    args = parser.parse_args()
    run_stacking(args.target, args.config, args.folds)


if __name__ == "__main__":
    main()
