"""
FusionModel v2 評估腳本

用法：
    python evaluate.py --target popularity --split test
    python evaluate.py --target meanScore  --split holdout_unknown
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fussion_training"))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

from dataset import AnimeDataset, denormalize_target
from meta_encoder import MetaEncoder
from model import FusionModel, make_model_config, apply_target_overrides


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, ids = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        preds.append(model(batch).cpu().numpy())
        ids.extend(batch["id"].cpu().tolist())
    return np.concatenate(preds), ids


def evaluate(target: str, split: str, config: dict, meta_encoder: MetaEncoder):
    config    = apply_target_overrides(config, target)   # 與 train 一致（架構/超參對齊）
    cfg_train = config["training"]
    cfg_out   = config["output"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir     = Path(cfg_out["run_dir"]) / cfg_out["run_id"] / target
    ckpt        = run_dir / "best_model.pt"
    scaler_path = run_dir / "target_scaler.json"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run train.py first.")

    with open(scaler_path) as f:
        target_scaler = json.load(f)

    ds = AnimeDataset(split, config, meta_encoder, target=target,
                      target_scaler=target_scaler)
    loader = DataLoader(
        ds, batch_size=cfg_train["batch_size"] * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    model = FusionModel(make_model_config(config, target)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    preds_norm, ids = _predict(model, loader, device)
    targets_norm    = ds.target_vals

    preds_orig   = denormalize_target(preds_norm,   target_scaler)
    targets_orig = denormalize_target(targets_norm, target_scaler)

    pred_df = pd.DataFrame({"id": ids, "pred": preds_orig, "target": targets_orig})
    pred_df.to_csv(run_dir / f"pred_{split}.csv", index=False)

    # 只在有真實標籤時計算 metrics（holdout targets 全為 0 表示未知）
    has_labels = not np.all(targets_norm == 0.0)
    if has_labels:
        mae      = float(np.mean(np.abs(preds_orig - targets_orig)))
        spearman = float(stats.spearmanr(preds_orig, targets_orig).correlation)
        if target == "popularity":
            log_p  = np.log1p(np.clip(preds_orig,   0, None))
            log_t  = np.log1p(np.clip(targets_orig, 0, None))
            ss_res = float(np.sum((log_t - log_p) ** 2))
            ss_tot = float(np.sum((log_t - np.mean(log_t)) ** 2))
            log_diff     = np.abs(log_p - log_t)
            factor_acc_2x = float(np.mean(log_diff < np.log(2)))
            metrics = {
                "target": target, "split": split, "n": len(preds_orig),
                "spearman_rho":  round(spearman, 4),
                "log_R2":        round(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4),
                "MAE":           round(mae, 4),
                "log_MAE":       round(float(np.mean(log_diff)), 4),
                "factor_acc_2x": round(factor_acc_2x, 4),
            }
        else:
            ss_res = float(np.sum((targets_orig - preds_orig) ** 2))
            ss_tot = float(np.sum((targets_orig - np.mean(targets_orig)) ** 2))
            acc_within_10pt = float(np.mean(np.abs(preds_orig - targets_orig) < 10))
            metrics = {
                "target": target, "split": split, "n": len(preds_orig),
                "spearman_rho":    round(spearman, 4),
                "R2":              round(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, 4),
                "MAE":             round(mae, 4),
                "acc_within_10pt": round(acc_within_10pt, 4),
            }
        print(f"\n{'='*50}")
        print(f"  {target} | {split}")
        for k, v in metrics.items():
            if k not in ("target", "split"):
                print(f"  {k:15s}: {v}")
        print("="*50)

        # merge into final_metrics.json
        final_path = run_dir / "final_metrics.json"
        summary = json.loads(final_path.read_text()) if final_path.exists() else \
                  {"target": target}
        summary[split] = {k: v for k, v in metrics.items()
                          if k not in ("target", "split", "n")}
        final_path.write_text(json.dumps(summary, indent=2))
        print(f"Merged   → {final_path}")
    else:
        print(f"\n{target} | {split}: no true labels, skipping metrics.")
        metrics = {"target": target, "split": split, "n": len(preds_orig)}

    print(f"Predictions → {run_dir}/pred_{split}.csv")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src_2/fussion_configs.yaml")
    parser.add_argument("--target", default=None, choices=["popularity", "meanScore"],
                        help="指定單一 target（預設跑 config 中所有 targets）")
    parser.add_argument("--split",  default="test",
                        choices=["train", "val", "test", "holdout_unknown"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    meta_encoder_path = Path(config["data"]["meta_encoder_path"])
    if not meta_encoder_path.exists():
        raise FileNotFoundError(
            f"MetaEncoder not found at {meta_encoder_path}. Run train.py first.")
    meta_encoder = MetaEncoder.load(str(meta_encoder_path))

    targets = [args.target] if args.target else config.get("targets", ["popularity"])
    for target in targets:
        evaluate(target, args.split, config, meta_encoder)


if __name__ == "__main__":
    main()
