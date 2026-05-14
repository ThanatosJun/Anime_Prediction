"""
Train FusionMLP + TextGNN + ImageGNN for a single target (popularity or meanScore).

Pipeline per batch:
    1. TextGNN(query_text, retrieved_text, mask) → enhanced_text
    2. ImageGNN(query_image, retrieved_image, mask) → enhanced_image
    3. FusionMLP(enhanced_text, enhanced_image, meta) → prediction

Output layout per target:
    results/fusion/{run_id}/{target}/
        best_model.pt        ← combined checkpoint {fusion_mlp, text_gnn, image_gnn}
        model_config.json    ← architecture config (FusionMLP + GNN params)
        target_scaler.json   ← normalisation params
        training_log.jsonl   ← per-epoch metrics
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.fussion_branch.fussion_training.dataset import FusionDataset
from src.fussion_branch.fussion_training.gnn import TextGNN, ImageGNN
from src.fussion_branch.fussion_training.model import FusionMLP
from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder
from src.fussion_branch.utilities.evaluate import compute_metrics, denormalize


def _build_target_scaler(
    meta_df: pd.DataFrame, target_col: str, log_transform: bool, winsor_pct: float | None = None
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


def _forward(
    batch: dict,
    model: FusionMLP,
    text_gnn: TextGNN,
    img_gnn: ImageGNN,
    device: torch.device,
) -> torch.Tensor:
    """Run GNN → FusionMLP forward pass for one batch.

    Caller is responsible for wrapping this in autocast when use_amp=True.
    """
    text  = batch["text_emb"].to(device)
    image = batch["image_emb"].to(device)
    meta  = batch["meta_feat"].to(device)
    r_txt = batch["ret_text"].to(device)
    r_img = batch["ret_image"].to(device)
    mask  = batch["ret_mask"].to(device)

    enh_text  = text_gnn(text, r_txt, mask)
    enh_image = img_gnn(image, r_img, mask)
    return model(enh_text, enh_image, meta)


def train_one_target(config: dict, target_col: str) -> dict:
    cfg_data  = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]
    cfg_out   = config["output"]
    log_transform = config["targets"][target_col]["log_transform"]
    winsor_pct    = config["targets"][target_col].get("winsor_pct", None)
    use_amp = cfg_train.get("mixed_precision", True) and torch.cuda.is_available()

    device = torch.device(
        cfg_train["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"\n[{target_col}] device={device}  log_transform={log_transform}  amp={use_amp}")

    # ── output dir ────────────────────────────────────────────────────────────
    out_dir = Path(cfg_out["results_dir"]) / cfg_out["run_id"] / target_col
    out_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(out_dir / "tb"))

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
    top_k_ids     = cfg_data.get("top_k_ids", 5)

    def make_dataset(split, apply_winsor: bool = False):
        return FusionDataset(
            split=split,
            encoder=encoder,
            meta_dir=cfg_data["fusion_meta_dir"],
            text_emb_dir=cfg_data["text_emb_dir"],
            rag_dir=cfg_data["rag_features_dir"],
            image_emb_dir=image_emb_dir,
            target_col=target_col,
            log_transform_target=log_transform,
            target_mean=scaler["mean"],
            target_std=scaler["std"],
            winsor_cap=scaler.get("winsor_cap") if apply_winsor else None,
            top_k_ids=top_k_ids,
        )

    train_ds = make_dataset("train", apply_winsor=True)
    val_ds   = make_dataset("val",   apply_winsor=False)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg_train["batch_size"], shuffle=False,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # ── models ────────────────────────────────────────────────────────────────
    gnn_num_layers = cfg_model.get("gnn_num_layers", 1)
    gnn_dropout    = cfg_model.get("gnn_dropout", 0.1)

    # FusionMLP only receives the keys it understands
    mlp_keys = {"text_proj", "image_proj", "meta_proj", "hidden_dims", "dropout"}
    mlp_cfg  = {k: v for k, v in cfg_model.items() if k in mlp_keys}

    text_gnn = TextGNN(num_layers=gnn_num_layers, dropout=gnn_dropout).to(device)
    img_gnn  = ImageGNN(num_layers=gnn_num_layers, dropout=gnn_dropout).to(device)
    model    = FusionMLP(
        text_dim=train_ds.text_dim,
        image_dim=train_ds.image_dim,
        meta_dim=train_ds.meta_dim,
        **mlp_cfg,
    ).to(device)

    # Save combined model config (FusionMLP + GNN params)
    full_cfg = {
        **model.get_config(),
        "gnn_num_layers": gnn_num_layers,
        "gnn_dropout":    gnn_dropout,
        "top_k_ids":      top_k_ids,
    }
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(full_cfg, f, indent=2)

    n_params = (
        sum(p.numel() for p in model.parameters()    if p.requires_grad) +
        sum(p.numel() for p in text_gnn.parameters() if p.requires_grad) +
        sum(p.numel() for p in img_gnn.parameters()  if p.requires_grad)
    )
    print(f"  Dims: text={train_ds.text_dim}  image={train_ds.image_dim} "
          f"({'real' if train_ds.use_image else 'zeros'})  meta={train_ds.meta_dim}")
    print(f"  GNN: num_layers={gnn_num_layers}  top_k={top_k_ids}")
    print(f"  Trainable params: {n_params:,}")

    criterion = nn.HuberLoss(delta=1.0)
    base_lr     = cfg_train["learning_rate"]
    gnn_lr      = base_lr * cfg_train.get("gnn_lr_factor", 1.0)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()),    "lr": base_lr},
            {"params": list(text_gnn.parameters()), "lr": gnn_lr},
            {"params": list(img_gnn.parameters()),  "lr": gnn_lr},
        ],
        weight_decay=cfg_train["weight_decay"],
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    scaler_amp = GradScaler("cuda") if use_amp else None

    # ── training loop ─────────────────────────────────────────────────────────
    warmup_epochs = cfg_train.get("warmup_epochs", 5)
    # target LRs per group (set by optimizer above)
    target_lrs    = [pg["lr"] for pg in optimizer.param_groups]
    best_val_mae  = float("inf")
    patience_cnt  = 0
    best_epoch    = 0
    log_path      = out_dir / "training_log.jsonl"
    ckpt_path     = out_dir / "best_model.pt"

    # save initial state as fallback
    _save_ckpt(ckpt_path, model, text_gnn, img_gnn)

    for epoch in range(1, cfg_train["epochs"] + 1):

        # linear LR warmup — respect per-group target LR
        if epoch <= warmup_epochs:
            for pg, target_lr in zip(optimizer.param_groups, target_lrs):
                pg["lr"] = target_lr * epoch / warmup_epochs

        # ── train ─────────────────────────────────────────────────────────────
        model.train(); text_gnn.train(); img_gnn.train()
        train_loss = 0.0
        for batch in train_loader:
            y = batch["target"].to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    pred = _forward(batch, model, text_gnn, img_gnn, device)
                    loss = criterion(pred, y)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(text_gnn.parameters()) + list(img_gnn.parameters()),
                    cfg_train["grad_clip"],
                )
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                pred = _forward(batch, model, text_gnn, img_gnn, device)
                loss = criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(text_gnn.parameters()) + list(img_gnn.parameters()),
                    cfg_train["grad_clip"],
                )
                optimizer.step()

            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        # ── validate ──────────────────────────────────────────────────────────
        val_mae, val_loss = _eval_loss_mae(
            model, text_gnn, img_gnn, val_loader, criterion, device, use_amp
        )
        current_lr     = optimizer.param_groups[0]["lr"]
        current_gnn_lr = optimizer.param_groups[1]["lr"]

        if epoch > warmup_epochs:
            plateau_scheduler.step(val_mae)

        writer.add_scalar("loss/train", train_loss,     epoch)
        writer.add_scalar("loss/val",   val_loss,       epoch)
        writer.add_scalar("MAE/val",    val_mae,        epoch)
        writer.add_scalar("lr/mlp",     current_lr,     epoch)
        writer.add_scalar("lr/gnn",     current_gnn_lr, epoch)

        log_entry = {
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "val_mae": round(val_mae, 4),
            "lr": current_lr, "gnn_lr": current_gnn_lr,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_MAE={val_mae:.4f}  lr={current_lr:.2e}")

        # ── early stopping ────────────────────────────────────────────────────
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            patience_cnt = 0
            _save_ckpt(ckpt_path, model, text_gnn, img_gnn)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg_train["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best={best_epoch}, val_MAE={best_val_mae:.4f})")
                break

    writer.close()
    print(f"  Best epoch: {best_epoch}  best val_MAE: {best_val_mae:.4f}")

    # ── final val evaluation ──────────────────────────────────────────────────
    _load_ckpt(ckpt_path, model, text_gnn, img_gnn, device)
    metrics_val = _full_eval(
        model, text_gnn, img_gnn, val_loader, scaler, target_col, device, use_amp, "val"
    )
    with open(out_dir / "metrics_val.json", "w") as f:
        json.dump(metrics_val, f, indent=2)

    print(f"  Outputs saved → {out_dir}")
    return {"val": metrics_val}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_ckpt(path, model, text_gnn, img_gnn):
    torch.save({
        "fusion_mlp": model.state_dict(),
        "text_gnn":   text_gnn.state_dict(),
        "image_gnn":  img_gnn.state_dict(),
    }, path)


def _load_ckpt(path, model, text_gnn, img_gnn, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["fusion_mlp"])
    text_gnn.load_state_dict(ckpt["text_gnn"])
    img_gnn.load_state_dict(ckpt["image_gnn"])


def _eval_loss_mae(model, text_gnn, img_gnn, loader, criterion, device, use_amp):
    model.eval(); text_gnn.eval(); img_gnn.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["target"].to(device)
            if use_amp:
                with autocast("cuda"):
                    pred = _forward(batch, model, text_gnn, img_gnn, device)
            else:
                pred = _forward(batch, model, text_gnn, img_gnn, device)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    mae      = float(np.mean(np.abs(y_true - y_pred)))
    val_loss = float(np.mean((y_true - y_pred) ** 2))
    return mae, val_loss


def _full_eval(model, text_gnn, img_gnn, loader, scaler, target_col, device, use_amp, split_name):
    model.eval(); text_gnn.eval(); img_gnn.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            if use_amp:
                with autocast("cuda"):
                    pred = _forward(batch, model, text_gnn, img_gnn, device)
            else:
                pred = _forward(batch, model, text_gnn, img_gnn, device)
            y_true_list.append(batch["target"].numpy())
            y_pred_list.append(pred.cpu().numpy())
    y_true_norm = np.concatenate(y_true_list)
    y_pred_norm = np.concatenate(y_pred_list)
    y_true = denormalize(y_true_norm, scaler)
    y_pred = denormalize(y_pred_norm, scaler)
    metrics = compute_metrics(y_true, y_pred, target_col)
    print(f"  [{split_name}] " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics
