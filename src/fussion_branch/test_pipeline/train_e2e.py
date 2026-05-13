"""
Train FusionMLPE2E for a single target (popularity or meanScore).

Key difference from train.py:
  - Text encoder backbone uses backbone_lr (much lower)
  - Batch contains tokenized text tensors instead of pre-computed embeddings
  - No text_emb parquet needed

Output layout: same as train.py under results_e2e/{run_id}/{target}/
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from fussion_branch.test_pipeline.dataset_e2e import FusionDatasetE2E
from fussion_branch.test_pipeline.model_e2e import FusionMLPE2E
from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder
from src.fussion_branch.utilities.evaluate import compute_metrics, denormalize


def _build_target_scaler(
    meta_df: pd.DataFrame, target_col: str, log_transform: bool, winsor_pct=None
) -> dict:
    y = meta_df[target_col].dropna().values.astype(np.float64)
    if log_transform:
        y = np.log1p(y)
    winsor_cap = None
    if winsor_pct is not None:
        winsor_cap = float(np.percentile(y, winsor_pct))
        y = np.clip(y, None, winsor_cap)
        print(f"  Winsorize {target_col}: cap={winsor_cap:.4f} (log space, {winsor_pct}th pct)")
    mean = float(y.mean())
    std  = float(y.std())
    return {"mean": mean, "std": max(std, 1e-8), "log_transform": log_transform,
            "winsor_cap": winsor_cap}


def train_one_target_e2e(config: dict, target_col: str) -> dict:
    cfg_data    = config["data"]
    cfg_enc     = config["text_encoder"]
    cfg_model   = config["model"]
    cfg_train   = config["training"]
    cfg_out     = config["output"]
    log_transform = config["targets"][target_col]["log_transform"]
    winsor_pct    = config["targets"][target_col].get("winsor_pct", None)
    use_amp = cfg_train.get("mixed_precision", True) and torch.cuda.is_available()

    device = torch.device(cfg_train["device"] if torch.cuda.is_available() else "cpu")
    print(f"\n[{target_col}] device={device}  log_transform={log_transform}  amp={use_amp}")

    # ── output dir ────────────────────────────────────────────────────────────
    out_dir = Path(cfg_out["results_dir"]) / cfg_out["run_id"] / target_col
    out_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ── tokenizer ─────────────────────────────────────────────────────────────
    encoder_name = cfg_enc["model_name"]
    max_length   = cfg_enc.get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    print(f"  Tokenizer: {encoder_name}  max_length={max_length}")

    # ── MetaEncoder ───────────────────────────────────────────────────────────
    encoder_path = cfg_data["meta_encoder_path"]
    if Path(encoder_path).exists():
        print(f"  Loading MetaEncoder from {encoder_path}")
        encoder = MetaEncoder.load(encoder_path)
    else:
        print("  Fitting MetaEncoder on training set…")
        meta_train_raw = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")
        rag_train_raw  = pd.read_parquet(f"{cfg_data['rag_features_dir']}/rag_features_train.parquet")
        encoder = MetaEncoder().fit(meta_train_raw, rag_train_raw)
        encoder.save(encoder_path)
        print(f"  MetaEncoder saved → {encoder_path}  (feature_dim={encoder.feature_dim})")

    # ── target scaler ─────────────────────────────────────────────────────────
    meta_train = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")
    scaler = _build_target_scaler(meta_train, target_col, log_transform, winsor_pct)
    with open(out_dir / "target_scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    # ── datasets & loaders ────────────────────────────────────────────────────
    image_emb_dir = cfg_data.get("image_emb_dir", "src/fussion_branch/embedding/image")

    def make_dataset(split, apply_winsor: bool = False):
        return FusionDatasetE2E(
            split=split,
            encoder=encoder,
            tokenizer=tokenizer,
            max_length=max_length,
            meta_dir=cfg_data["fusion_meta_dir"],
            rag_dir=cfg_data["rag_features_dir"],
            image_emb_dir=image_emb_dir,
            target_col=target_col,
            log_transform_target=log_transform,
            target_mean=scaler["mean"],
            target_std=scaler["std"],
            winsor_cap=scaler.get("winsor_cap") if apply_winsor else None,
        )

    train_ds = make_dataset("train", apply_winsor=True)
    val_ds   = make_dataset("val",   apply_winsor=False)

    # fewer workers to avoid tokenizer forking issues
    num_workers = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg_train["batch_size"], shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model = FusionMLPE2E(
        encoder_name=encoder_name,
        freeze_layers=cfg_enc["freeze_layers"],
        image_dim=train_ds.image_dim,
        meta_dim=train_ds.meta_dim,
        **cfg_model,
    ).to(device)
    model.save_config(str(out_dir / "model_config.json"))

    n_backbone = sum(p.numel() for p in model.backbone_parameters())
    n_head     = sum(p.numel() for p in model.head_parameters())
    print(f"  Trainable: backbone={n_backbone:,}  head={n_head:,}  total={n_backbone+n_head:,}")

    # ── differential optimizer ────────────────────────────────────────────────
    backbone_lr = cfg_enc.get("backbone_lr", 1e-5)
    head_lr     = cfg_train["learning_rate"]
    criterion   = nn.HuberLoss(delta=1.0)
    optimizer   = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": backbone_lr},
            {"params": model.head_parameters(),     "lr": head_lr},
        ],
        weight_decay=cfg_train["weight_decay"],
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7,
    )
    scaler_amp = GradScaler("cuda") if use_amp else None

    # ── training loop ─────────────────────────────────────────────────────────
    warmup_epochs = cfg_train.get("warmup_epochs", 5)
    best_val_mae  = float("inf")
    patience_cnt  = 0
    best_epoch    = 0
    log_path      = out_dir / "training_log.jsonl"
    ckpt_path     = out_dir / "best_model.pt"

    torch.save(model.state_dict(), ckpt_path)

    for epoch in range(1, cfg_train["epochs"] + 1):

        # linear warmup (head LR only; backbone LR stays at configured value)
        if epoch <= warmup_epochs:
            optimizer.param_groups[1]["lr"] = head_lr * epoch / warmup_epochs

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_emb      = batch["image_emb"].to(device)
            meta_feat      = batch["meta_feat"].to(device)
            y              = batch["target"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    pred = model(input_ids, attention_mask, image_emb, meta_feat)
                    loss = criterion(pred, y)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg_train["grad_clip"])
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                pred = model(input_ids, attention_mask, image_emb, meta_feat)
                loss = criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg_train["grad_clip"])
                optimizer.step()

            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        # ── validate ──────────────────────────────────────────────────────────
        val_mae, val_loss = _eval_loss_mae(model, val_loader, criterion, device, use_amp)
        current_lr_head = optimizer.param_groups[1]["lr"]

        if epoch > warmup_epochs:
            plateau_scheduler.step(val_mae)

        writer.add_scalar("loss/train",  train_loss, epoch)
        writer.add_scalar("loss/val",    val_loss,   epoch)
        writer.add_scalar("MAE/val",     val_mae,    epoch)
        writer.add_scalar("lr/backbone", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("lr/head",     current_lr_head, epoch)

        log_entry = {
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "val_mae": round(val_mae, 4),
            "lr_head": current_lr_head,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_MAE={val_mae:.4f}  lr_head={current_lr_head:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg_train["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best={best_epoch}, val_MAE={best_val_mae:.4f})")
                break

    writer.close()
    print(f"  Best epoch: {best_epoch}  best val_MAE: {best_val_mae:.4f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    best_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)

    metrics_val = _full_eval(model, val_loader, scaler, target_col, device, use_amp, "val")
    with open(out_dir / "metrics_val.json", "w") as f:
        json.dump(metrics_val, f, indent=2)

    print(f"  Outputs saved → {out_dir}")
    return {"val": metrics_val}


# ── helpers ───────────────────────────────────────────────────────────────────

def _eval_loss_mae(model, loader, criterion, device, use_amp):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_emb      = batch["image_emb"].to(device)
            meta_feat      = batch["meta_feat"].to(device)
            y              = batch["target"].to(device)
            if use_amp:
                with autocast("cuda"):
                    pred = model(input_ids, attention_mask, image_emb, meta_feat)
            else:
                pred = model(input_ids, attention_mask, image_emb, meta_feat)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    mae      = float(np.mean(np.abs(y_true - y_pred)))
    val_loss = float(np.mean((y_true - y_pred) ** 2))
    return mae, val_loss


def _full_eval(model, loader, scaler, target_col, device, use_amp, split_name):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_emb      = batch["image_emb"].to(device)
            meta_feat      = batch["meta_feat"].to(device)
            if use_amp:
                with autocast("cuda"):
                    pred = model(input_ids, attention_mask, image_emb, meta_feat)
            else:
                pred = model(input_ids, attention_mask, image_emb, meta_feat)
            y_true_list.append(batch["target"].numpy())
            y_pred_list.append(pred.cpu().numpy())
    y_true_norm = np.concatenate(y_true_list)
    y_pred_norm = np.concatenate(y_pred_list)
    y_true = denormalize(y_true_norm, scaler)
    y_pred = denormalize(y_pred_norm, scaler)
    metrics = compute_metrics(y_true, y_pred, target_col)
    print(f"  [{split_name}] " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics
