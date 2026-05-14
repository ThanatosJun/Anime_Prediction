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
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.fussion_branch.fussion_training.dataset import FusionDataset
from src.fussion_branch.fussion_training.gnn import TextGNN, ImageGNN
from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder
from src.fussion_branch.fussion_training.model import FusionMLP
from src.fussion_branch.utilities.config import load_config
from src.fussion_branch.utilities.evaluate import compute_metrics, denormalize
from src.fussion_branch.utilities.summarize_experiments import collect


def _forward(batch, model, text_gnn, img_gnn, device):
    text  = batch["text_emb"].to(device)
    image = batch["image_emb"].to(device)
    meta  = batch["meta_feat"].to(device)
    r_txt = batch["ret_text"].to(device)
    r_img = batch["ret_image"].to(device)
    mask  = batch["ret_mask"].to(device)
    return model(text_gnn(text, r_txt, mask), img_gnn(image, r_img, mask), meta)


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

    with open(model_config_path) as f:
        model_cfg = json.load(f)
    with open(scaler_path) as f:
        scaler = json.load(f)

    # ── reconstruct models from saved config ──────────────────────────────────
    gnn_num_layers = model_cfg.get("gnn_num_layers", 1)
    gnn_dropout    = model_cfg.get("gnn_dropout", 0.1)
    top_k_ids      = model_cfg.get("top_k_ids", 5)

    mlp_keys = {"text_dim", "image_dim", "meta_dim", "text_proj", "image_proj",
                "meta_proj", "hidden_dims", "dropout"}
    mlp_cfg  = {k: v for k, v in model_cfg.items() if k in mlp_keys}

    model    = FusionMLP(**mlp_cfg).to(device)
    text_gnn = TextGNN(num_layers=gnn_num_layers, dropout=gnn_dropout).to(device)
    img_gnn  = ImageGNN(num_layers=gnn_num_layers, dropout=gnn_dropout).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "fusion_mlp" in ckpt:
        model.load_state_dict(ckpt["fusion_mlp"])
        text_gnn.load_state_dict(ckpt["text_gnn"])
        img_gnn.load_state_dict(ckpt["image_gnn"])
    else:
        model.load_state_dict(ckpt)

    model.eval(); text_gnn.eval(); img_gnn.eval()
    encoder = MetaEncoder.load(encoder_path)

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
        top_k_ids=top_k_ids,
    )

    num_workers = min(4, os.cpu_count() or 1)
    test_loader = DataLoader(test_ds, batch_size=cfg_train["batch_size"], shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # ── evaluate ──────────────────────────────────────────────────────────────
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            if use_amp:
                with autocast("cuda"):
                    pred = _forward(batch, model, text_gnn, img_gnn, device)
            else:
                pred = _forward(batch, model, text_gnn, img_gnn, device)
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
