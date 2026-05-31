"""
FusionModel v2 訓練主程式

用法：
    python train.py                          # 依 fussion_configs.yaml 訓練
    python train.py --config path/to/cfg.yaml
    python train.py --target popularity      # 只訓練指定 target
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fussion_training"))

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AnimeDataset, denormalize_target
from meta_encoder import MetaEncoder


def set_seed(seed: int):
    """固定所有隨機源（weight init / shuffle / dropout），讓實驗可重現、變因對齊。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False   # 關閉非確定性 kernel 選擇
from model import FusionModel, make_model_config


# ── Loss functions ────────────────────────────────────────────────────────────

class LogCoshLoss(nn.Module):
    _LOG2 = math.log(2)

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Numerically stable: log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
        # Avoids cosh overflow for large |diff| unlike the naive formulation.
        diff = pred - target
        loss = diff.abs() + torch.log1p(torch.exp(-2.0 * diff.abs())) - self._LOG2
        return loss.mean() if self.reduction == "mean" else loss


# ── SAM Optimizer ─────────────────────────────────────────────────────────────

class SAM:
    """Sharpness-Aware Minimization wrapper around a base optimizer."""

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.rho            = rho
        self.param_groups   = base_optimizer.param_groups
        self._e_w: dict     = {}

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)
                self._e_w[p] = e_w
        if zero_grad:
            self.zero_grad()

    def second_step(self, zero_grad: bool = False):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p not in self._e_w:
                        continue
                    p.sub_(self._e_w.pop(p))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


# ── Training helpers ──────────────────────────────────────────────────────────

def _get_criterion(target: str, reduction: str = "mean"):
    if target == "meanScore":
        return LogCoshLoss(reduction=reduction)
    return nn.HuberLoss(delta=1.0, reduction=reduction)


def _train_epoch(model, loader, criterion_none, optimizer, device, epoch, epochs, use_amp):
    """criterion_none: reduction='none'，由此函式自行加權後取 mean。"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False, ncols=90)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        y = batch["target"]
        w = batch.get("weight", torch.ones_like(y))   # [batch]

        # SAM first step
        with autocast("cuda", enabled=use_amp):
            pred = model(batch)
            loss = (criterion_none(pred, y) * w).mean()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second step
        with autocast("cuda", enabled=use_amp):
            loss2 = (criterion_none(model(batch), y) * w).mean()
        loss2.backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item() * len(y)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device, use_amp: bool = False,
                target_scaler: dict = None):
    model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        y = batch["target"]
        with autocast("cuda", enabled=use_amp):
            pred = model(batch)
        total_loss += criterion(pred, y).item() * len(y)
        preds_all.append(pred.cpu().numpy())
        targets_all.append(y.cpu().numpy())

    preds   = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    if target_scaler is not None:
        po = denormalize_target(preds,   target_scaler)
        to = denormalize_target(targets, target_scaler)
        mae = float(np.mean(np.abs(po - to)))
    else:
        mae = float(np.mean(np.abs(preds - targets)))
    return total_loss / len(loader.dataset), mae


@torch.no_grad()
def _compute_final_metrics(target, run_dir, datasets, config, device, use_amp):
    """訓練結束後，載入 best checkpoint，計算 train/val 全量 metrics。"""
    model = FusionModel(make_model_config(config, target)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model.eval()
    batch_size = config["training"]["batch_size"] * 2

    results = {}
    for split_name, ds in datasets.items():
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
        preds_n, targets_n = [], []
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with autocast("cuda", enabled=use_amp):
                pred = model(batch)
            preds_n.append(pred.cpu().numpy())
            targets_n.append(batch["target"].cpu().numpy())

        preds_n   = np.concatenate(preds_n)
        targets_n = np.concatenate(targets_n)
        po = denormalize_target(preds_n,   ds.target_scaler)
        to = denormalize_target(targets_n, ds.target_scaler)

        mae      = float(np.mean(np.abs(po - to)))
        spearman = float(scipy_stats.spearmanr(po, to).correlation)
        ss_res   = float(np.sum((to - po) ** 2))
        ss_tot   = float(np.sum((to - np.mean(to)) ** 2))

        if target == "popularity":
            lp = np.log1p(np.clip(po, 0, None))
            lt = np.log1p(np.clip(to, 0, None))
            log_diff = np.abs(lp - lt)
            sr = float(np.sum((lt - lp) ** 2))
            st = float(np.sum((lt - np.mean(lt)) ** 2))
            m = {
                "spearman_rho":  round(spearman, 4),
                "log_R2":        round(1.0 - sr / st if st > 0 else 0.0, 4),
                "MAE":           round(mae, 4),
                "log_MAE":       round(float(np.mean(log_diff)), 4),
                "factor_acc_2x": round(float(np.mean(log_diff < np.log(2))), 4),
            }
        else:
            m = {
                "spearman_rho":    round(spearman, 4),
                "R2":              round(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4),
                "MAE":             round(mae, 4),
                "acc_within_10pt": round(float(np.mean(np.abs(po - to) < 10)), 4),
            }
        results[split_name] = m

    return results


# ── Train single target ───────────────────────────────────────────────────────

def train_target(target: str, config: dict, meta_encoder: MetaEncoder):
    cfg_train = config["training"]
    cfg_out   = config["output"]
    seed      = config.get("seed", 42)
    set_seed(seed)   # 每個 target 從相同 seed 起跑，變因對齊、可重現
    device    = torch.device(cfg_train.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg_out["run_dir"]) / cfg_out["run_id"] / target
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Target: {target}  |  Device: {device}")
    print(f"Run dir: {run_dir}")
    print('='*60)

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = AnimeDataset("train", config, meta_encoder, target=target)
    val_ds   = AnimeDataset("val",   config, meta_encoder, target=target,
                            target_scaler=train_ds.target_scaler)

    num_workers = min(4, os.cpu_count() or 1)
    g = torch.Generator().manual_seed(seed)   # 固定 shuffle 順序
    train_loader = DataLoader(
        train_ds, batch_size=cfg_train["batch_size"],
        shuffle=True, num_workers=num_workers, pin_memory=True,
        persistent_workers=True, drop_last=False, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_train["batch_size"] * 2,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=True,
    )

    # ── model ─────────────────────────────────────────────────────────────
    use_amp = cfg_train.get("mixed_precision", True) and torch.cuda.is_available()
    model   = FusionModel(make_model_config(config, target)).to(device)
    criterion      = _get_criterion(target, reduction="mean")   # eval
    criterion_none = _get_criterion(target, reduction="none")   # train（加權用）
    base_opt  = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_train["lr"],
        weight_decay=cfg_train["weight_decay"],
    )
    optimizer = SAM(base_opt, rho=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        base_opt, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )

    print(f"AMP: {use_amp}  |  num_workers: {num_workers}")

    # ── training loop ─────────────────────────────────────────────────────
    patience   = cfg_train.get("patience", 5)
    best_val   = float("inf")
    best_epoch = 0
    no_improve = 0
    history    = []

    for epoch in range(1, cfg_train["epochs"] + 1):
        train_loss        = _train_epoch(model, train_loader, criterion_none, optimizer, device, epoch, cfg_train["epochs"], use_amp)
        val_loss, val_mae = _eval_epoch(model, val_loader, criterion, device, use_amp,
                                        target_scaler=train_ds.target_scaler)
        scheduler.step(val_loss)
        current_lr = base_opt.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6), "val_mae": round(val_mae, 6),
            "lr": current_lr,
        })
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}"
              f" | mae={val_mae:.4f} | lr={current_lr:.2e}")

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            with open(run_dir / "target_scaler.json", "w") as f:
                json.dump(train_ds.target_scaler, f, indent=2)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    notes    = cfg_out.get("notes", "")
    run_id   = cfg_out["run_id"]

    print(f"\nBest epoch: {best_epoch} | Best val loss: {best_val:.4f}")
    with open(run_dir / "history.json", "w") as f:
        json.dump({"run_id": run_id, "target": target, "notes": notes,
                   "best_epoch": best_epoch, "best_val_loss": best_val,
                   "history": history}, f, indent=2)

    # ── Final metrics on train + val ──────────────────────────────────────
    print("Computing final metrics...")
    final_metrics = _compute_final_metrics(
        target, run_dir,
        {"train": train_ds, "val": val_ds},
        config, device, use_amp,
    )
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump({"run_id": run_id, "target": target, "notes": notes,
                   "best_epoch": best_epoch, **final_metrics}, f, indent=2)
    print("  train: " + "  ".join(f"{k}={v}" for k, v in final_metrics["train"].items()))
    print("  val:   " + "  ".join(f"{k}={v}" for k, v in final_metrics["val"].items()))

    return best_val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src_2/fussion_configs.yaml")
    parser.add_argument("--target", default=None,
                        help="指定單一 target（預設跑 config 中所有 targets）")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Fit MetaEncoder on training data
    meta_suffix = config["data"].get("meta_suffix", "_v2")
    train_csv   = Path(config["data"]["meta_dir"]) / f"fusion_meta_clean_train{meta_suffix}.csv"
    train_df    = pd.read_csv(train_csv)

    meta_encoder_path = Path(config["data"]["meta_encoder_path"])
    if meta_encoder_path.exists():
        print(f"Loading MetaEncoder from {meta_encoder_path}")
        meta_encoder = MetaEncoder.load(str(meta_encoder_path))
    else:
        print("Fitting MetaEncoder on training set...")
        meta_encoder = MetaEncoder().fit(train_df)
        meta_encoder.save(str(meta_encoder_path))
        print(f"MetaEncoder saved → {meta_encoder_path}  (feature_dim={meta_encoder.feature_dim})")

    if not config.get("train_separately", True):
        raise NotImplementedError("train_separately: false (multi-head) is not yet implemented.")

    targets = [args.target] if args.target else config.get("targets", ["popularity"])
    results = {}
    for target in targets:
        results[target] = train_target(target, config, meta_encoder)

    print("\n=== Training Summary ===")
    for t, v in results.items():
        print(f"  {t}: best val loss = {v:.4f}")


if __name__ == "__main__":
    main()
