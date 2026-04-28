"""
Scan .exp/fussion/results/ and compile all experiment metrics into a single CSV.

Output: .exp/fussion/experiments_summary.csv

Columns:
    run_id, target, dropout, hidden_dims, log_transform,
    val_Spearman_rho, val_R2, val_MAE, val_log_MAE,
    test_Spearman_rho, test_R2, test_MAE, test_log_MAE

Usage:
    python -m src.fussion_branch.utilities.summarize_experiments
"""
import json
from pathlib import Path

import pandas as pd
import yaml

RESULTS_DIR = Path(".exp/fussion/results")
OUT_CSV     = Path(".exp/fussion/experiments_summary.csv")


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def collect() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        run_cfg = _load_yaml(run_dir / "config.yaml")

        for target_dir in sorted(run_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            target = target_dir.name

            model_cfg  = _load_json(target_dir / "model_config.json")
            scaler     = _load_json(target_dir / "target_scaler.json")
            val_m      = _load_json(target_dir / "metrics_val.json")
            test_m     = _load_json(target_dir / "metrics_test.json")

            training = run_cfg.get("training", {})
            data     = run_cfg.get("data", {})

            row = {
                "run_id":           run_id,
                "target":           target,
                "notes":            run_cfg.get("output", {}).get("notes", ""),
                "dataset":          Path(data.get("fusion_meta_dir", "")).name or data.get("fusion_meta_dir", ""),
                "learning_rate":    training.get("learning_rate"),
                "weight_decay":     training.get("weight_decay"),
                "dropout":          model_cfg.get("dropout"),
                "hidden_dims":      str(model_cfg.get("hidden_dims")),
                "log_transform":    scaler.get("log_transform"),
            }
            for k, v in val_m.items():
                row[f"val_{k}"] = v
            for k, v in test_m.items():
                row[f"test_{k}"] = v

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = collect()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows → {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
