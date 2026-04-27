"""
Train FusionMLP for a single target (popularity or meanScore).

Output layout per target:
    results/fusion/{run_id}/{target}/
        best_model.pt        ← best val checkpoint
        target_scaler.json
        metrics_val.json
        metrics_test.json
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.fussion_branch.fussion_training.dataset import FusionDataset
from src.fussion_branch.utilities.evaluate import compute_metrics, denormalize
from src.fussion_branch.RAG.meta_encoder import MetaEncoder
from src.fussion_branch.fussion_training.model import FusionMLP


def _build_target_scaler(meta_df: pd.DataFrame, target_col: str, log_transform: bool) -> dict:
    y = meta_df[target_col].dropna().values.astype(np.float64)
    if log_transform:
        y = np.log1p(y)
    mean = float(y.mean())
    std  = float(y.std())
    return {"mean": mean, "std": max(std, 1e-8), "log_transform": log_transform}


def train_one_target(config: dict, target_col: str) -> dict:
    cfg_data  = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]
    cfg_out   = config["output"]
    log_transform = config["targets"][target_col]["log_transform"]

    device = torch.device(
        cfg_train["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"\n[{target_col}] device={device}  log_transform={log_transform}")

    # ── directories ───────────────────────────────────────────────────────────
    out_dir = Path(cfg_out["results_dir"]) / cfg_out["run_id"] / target_col
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ── MetaEncoder (fit once, reused for all splits) ─────────────────────────
    encoder_path = cfg_data["meta_encoder_path"]
    if Path(encoder_path).exists():
        print(f"  Loading MetaEncoder from {encoder_path}")
        encoder = MetaEncoder.load(encoder_path)
    else:
        print("  Fitting MetaEncoder on training set…")
        meta_train = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")
        rag_train  = pd.read_parquet(f"{cfg_data['rag_features_dir']}/rag_features_train.parquet")
        encoder = MetaEncoder(top_studios=cfg_data["top_studios"]).fit(meta_train, rag_train)
        encoder.save(encoder_path)
        print(f"  MetaEncoder saved → {encoder_path}  (feature_dim={encoder.feature_dim})")

    # ── target scaler ─────────────────────────────────────────────────────────
    meta_train = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")
    scaler = _build_target_scaler(meta_train, target_col, log_transform)
    with open(out_dir / "target_scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    # ── datasets & loaders ────────────────────────────────────────────────────
    image_emb_path = cfg_data.get("image_emb_path", "data/processed/image_embeddings.parquet")

    def make_dataset(split):
        return FusionDataset(
            split=split,
            encoder=encoder,
            meta_dir=cfg_data["fusion_meta_dir"],
            text_emb_dir=cfg_data["text_emb_dir"],
            rag_dir=cfg_data["rag_features_dir"],
            image_emb_path=image_emb_path,
            target_col=target_col,
            log_transform_target=log_transform,
            target_mean=scaler["mean"],
            target_std=scaler["std"],
        )

    train_ds = make_dataset("train")
    val_ds   = make_dataset("val")
    test_ds  = make_dataset("test")

    train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg_train["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg_train["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # ── model (input_dim from dataset to auto-include image if available) ─────
    input_dim = train_ds.feature_dim   # text(384) + image(1024 or 0) + meta_rag(~155)
    model = FusionMLP(
        input_dim=input_dim,
        hidden_dims=cfg_model["hidden_dims"],
        dropout=cfg_model["dropout"],
    ).to(device)
    use_img = train_ds.use_image
    print(f"  Input dim: {input_dim}  (text=384, image={'1024' if use_img else '0 (zeros)'}, meta_rag={encoder.feature_dim})")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_mae  = float("inf")
    patience_cnt  = 0
    best_epoch    = 0

    for epoch in range(1, cfg_train["epochs"] + 1):
        # train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_train["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        # validate
        val_mae, val_loss = _eval_loss_mae(model, val_loader, criterion, scaler, device)
        scheduler.step(val_mae)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("MAE/val",    val_mae,    epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_MAE={val_mae:.4f}")

        # early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg_train["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch}, val_MAE={best_val_mae:.4f})")
                break

    writer.close()
    print(f"  Best epoch: {best_epoch}  best val_MAE: {best_val_mae:.4f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    metrics_val  = _full_eval(model, val_loader,  scaler, target_col, meta_train, device, "val")
    metrics_test = _full_eval(model, test_loader, scaler, target_col, meta_train, device, "test")

    with open(out_dir / "metrics_val.json",  "w") as f:
        json.dump(metrics_val,  f, indent=2)
    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump(metrics_test, f, indent=2)

    return {"val": metrics_val, "test": metrics_test}


# ── helpers ───────────────────────────────────────────────────────────────────

def _eval_loss_mae(model, loader, criterion, scaler, device):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_true_raw = denormalize(y_true, scaler)
    y_pred_raw = denormalize(y_pred, scaler)
    mae      = float(np.mean(np.abs(y_true_raw - y_pred_raw)))
    val_loss = float(np.mean((y_true - y_pred) ** 2))
    return mae, val_loss


def _full_eval(model, loader, scaler, target_col, train_meta_df, device, split_name):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y_true_list.append(batch["target"].numpy())
            y_pred_list.append(model(x).cpu().numpy())
    y_true_norm = np.concatenate(y_true_list)
    y_pred_norm = np.concatenate(y_pred_list)
    y_true = denormalize(y_true_norm, scaler)
    y_pred = denormalize(y_pred_norm, scaler)
    metrics = compute_metrics(y_true, y_pred, target_col, train_meta_df)
    print(f"  [{split_name}] " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics
