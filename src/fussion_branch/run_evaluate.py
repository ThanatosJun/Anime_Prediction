"""
Final evaluation on the test set.

Run this ONCE after training is complete — test set should never be seen
during training or hyperparameter tuning.

Usage:
    python -m src.fussion_branch.run_evaluate
    python -m src.fussion_branch.run_evaluate --target popularity
    python -m src.fussion_branch.run_evaluate --target meanScore
    python -m src.fussion_branch.run_evaluate --target both
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.fussion_branch.fussion_training.dataset import FusionDataset
from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder
from src.fussion_branch.fussion_training.model import FusionMLP
from src.fussion_branch.utilities.config import load_config
from src.fussion_branch.utilities.evaluate import compute_metrics, denormalize
from src.fussion_branch.utilities.summarize_experiments import collect


def evaluate_target(config: dict, target_col: str):
    cfg_data  = config["data"]
    cfg_train = config["training"]
    cfg_out   = config["output"]
    use_amp   = cfg_train.get("mixed_precision", True) and torch.cuda.is_available()

    device  = torch.device(cfg_train["device"] if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg_out["results_dir"]) / cfg_out["run_id"] / target_col

    # ── load artifacts ────────────────────────────────────────────────────────
    model_config_path = out_dir / "model_config.json"
    checkpoint_path   = out_dir / "best_model.pt"
    scaler_path       = out_dir / "target_scaler.json"
    encoder_path      = cfg_data["meta_encoder_path"]

    for p in [model_config_path, checkpoint_path, scaler_path, encoder_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required file: {p}\nRun training first.")

    model   = FusionMLP.load(str(model_config_path), str(checkpoint_path), map_location=device)
    model   = model.to(device).eval()
    encoder = MetaEncoder.load(encoder_path)

    with open(scaler_path) as f:
        scaler = json.load(f)

    # ── test dataset ──────────────────────────────────────────────────────────
    image_emb_dir = cfg_data.get("image_emb_dir", "src/fussion_branch/embedding/image")
    log_transform = config["targets"][target_col]["log_transform"]

    test_ds = FusionDataset(
        split="test",
        encoder=encoder,
        meta_dir=cfg_data["fusion_meta_dir"],
        text_emb_dir=cfg_data["text_emb_dir"],
        rag_dir=cfg_data["rag_features_dir"],
        image_emb_dir=image_emb_dir,
        target_col=target_col,
        log_transform_target=log_transform,
        target_mean=scaler["mean"],
        target_std=scaler["std"],
    )

    num_workers = min(4, os.cpu_count() or 1)
    test_loader = DataLoader(test_ds, batch_size=cfg_train["batch_size"], shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── evaluate ──────────────────────────────────────────────────────────────
    meta_train = pd.read_csv(f"{cfg_data['fusion_meta_dir']}/fusion_meta_clean_train.csv")

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            if use_amp:
                with autocast("cuda"):
                    pred = model(x)
            else:
                pred = model(x)
            y_true_list.append(batch["target"].numpy())
            y_pred_list.append(pred.cpu().numpy())

    y_true = denormalize(np.concatenate(y_true_list), scaler)
    y_pred = denormalize(np.concatenate(y_pred_list), scaler)
    metrics = compute_metrics(y_true, y_pred, target_col)

    print(f"[{target_col}] test: " + "  ".join(f"{k}={v}" for k, v in metrics.items()))

    with open(out_dir / "metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {out_dir / 'metrics_test.json'}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["popularity", "meanScore", "both"],
        default=None,
        help="Override active_targets in config",
    )
    parser.add_argument(
        "--config",
        default="src/fussion_branch/configs/fusion_config.yaml",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override run_id in config (e.g. --run-id 02)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.run_id is not None:
        config["output"]["run_id"] = args.run_id
    if args.target is not None:
        targets = ["popularity", "meanScore"] if args.target == "both" else [args.target]
    else:
        targets = config.get("active_targets", ["popularity", "meanScore"])

    all_metrics = {}
    for target in targets:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {target}")
        print(f"{'='*60}")
        all_metrics[target] = evaluate_target(config, target)

    print("\n" + "="*60)
    print("  Final Test Metrics")
    print("="*60)
    print(json.dumps(all_metrics, indent=2))

    from pathlib import Path
    out_csv = Path(".exp/fussion/experiments_summary.csv")
    collect().to_csv(out_csv, index=False)
    print(f"\n[summary] updated → {out_csv}")


if __name__ == "__main__":
    main()
