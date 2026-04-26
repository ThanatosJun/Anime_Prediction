"""
Quick smoke-test for the fusion training pipeline.
Trains on 1000 samples / 300 val, 5 epochs, popularity only.

Usage:
    conda activate animeprediction
    python test_train.py
"""
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.fussion_branch.config import load_config
from src.fussion_branch.dataset import FusionDataset
from src.fussion_branch.evaluate import compute_metrics, denormalize
from src.fussion_branch.meta_encoder import MetaEncoder
from src.fussion_branch.model import FusionMLP

N_TRAIN  = 1000
N_VAL    = 300
EPOCHS   = 5
TARGET   = "popularity"


def main():
    config = load_config()

    # override to keep test fast
    cfg = copy.deepcopy(config)
    cfg["training"]["epochs"]                   = EPOCHS
    cfg["training"]["early_stopping_patience"]  = EPOCHS
    cfg["training"]["batch_size"]               = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  target={TARGET}  n_train={N_TRAIN}  n_val={N_VAL}  epochs={EPOCHS}")

    # ── MetaEncoder: fit on full training set for correct vocab ───────────────
    meta_train = pd.read_csv("data/fussion/fusion_meta_train.csv")
    rag_train  = pd.read_parquet("artifacts/rag_features_train.parquet")
    encoder = MetaEncoder(top_studios=cfg["data"]["top_studios"]).fit(meta_train, rag_train)
    print(f"MetaEncoder feature_dim={encoder.feature_dim}")

    # ── target scaler ─────────────────────────────────────────────────────────
    log_transform = cfg["targets"][TARGET]["log_transform"]
    y = meta_train[TARGET].dropna().values.astype(np.float64)
    if log_transform:
        y = np.log1p(y)
    t_mean, t_std = float(y.mean()), float(max(y.std(), 1e-8))

    # ── datasets (full load, then Subset) ─────────────────────────────────────
    def make_ds(split):
        return FusionDataset(
            split=split,
            encoder=encoder,
            meta_dir=cfg["data"]["fusion_meta_dir"],
            text_emb_dir=cfg["data"]["text_emb_dir"],
            rag_dir=cfg["data"]["rag_features_dir"],
            image_emb_path=cfg["data"]["image_emb_path"],
            target_col=TARGET,
            log_transform_target=log_transform,
            target_mean=t_mean,
            target_std=t_std,
        )

    train_ds = Subset(make_ds("train"), range(N_TRAIN))
    val_ds   = Subset(make_ds("val"),   range(N_VAL))

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2)

    # ── model ─────────────────────────────────────────────────────────────────
    input_dim = train_ds.dataset.feature_dim
    model = FusionMLP(
        input_dim=input_dim,
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    print(f"FusionMLP  input_dim={input_dim}  params={sum(p.numel() for p in model.parameters()):,}")

    scaler    = {"mean": t_mean, "std": t_std, "log_transform": log_transform}
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y_b = batch["features"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= N_TRAIN

        # val MAE in original scale
        model.eval()
        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                y_true_list.append(batch["target"].numpy())
                y_pred_list.append(model(batch["features"].to(device)).cpu().numpy())
        y_true = denormalize(np.concatenate(y_true_list), scaler)
        y_pred = denormalize(np.concatenate(y_pred_list), scaler)
        val_mae = float(np.mean(np.abs(y_true - y_pred)))

        print(f"  epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_MAE={val_mae:.1f}")

    # ── final metrics ─────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred, TARGET, meta_train)
    print("\n=== val metrics (subset) ===")
    print(json.dumps(metrics, indent=2))
    print("\nPipeline OK")


if __name__ == "__main__":
    main()
