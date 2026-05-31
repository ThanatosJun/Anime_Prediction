"""
FusionModel v2 Hyperparameter Search

搜尋空間（其餘設定固定，繼承 fussion_configs.yaml）：
  dropout      : [0.3, 0.4, 0.5]
  weight_decay : [1e-4, 5e-4, 1e-3]
  batch_size   : [256, 512]

固定不變：
  lr=0.001, image_mode=cover_banner_yolo, use_rag=true,
  TrendHead enabled (pop+score), hidden_dims=[256,128]

Run IDs : 04 ~ 09
結果摘要: src_2/runs/hp_summary.json
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fussion_training"))

import pandas as pd
import yaml

from meta_encoder import MetaEncoder
from train import train_target

# ── Search space ──────────────────────────────────────────────────────────────
# 每欄：(run_id, dropout, weight_decay, batch_size)
SEARCH_SPACE = [
    ("04", 0.3, 1e-4, 512),   # Run02 基準，只放大 batch
    ("05", 0.4, 5e-4, 512),   # 中等正則 + 大 batch
    ("06", 0.5, 1e-4, 512),   # 高 dropout only
    ("07", 0.3, 1e-3, 512),   # 高 weight_decay only
    ("08", 0.5, 5e-4, 512),   # 高 dropout + 中 wd
    ("09", 0.5, 1e-3, 512),   # 全強正則（= Run03 設定）
]

CONFIG_PATH = "src_2/fussion_configs.yaml"
SUMMARY_PATH = Path("src_2/runs/hp_summary.json")


def _make_run_notes(run_id, dropout, weight_decay, batch_size):
    return (f"hp Run{run_id}: dropout={dropout}, wd={weight_decay:.0e}, "
            f"batch={batch_size}; TrendHead(pop+score); baseline=Run02")


def main():
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    # MetaEncoder fit once，所有 run 共用
    meta_suffix = base_config["data"].get("meta_suffix", "_v2")
    train_csv   = Path(base_config["data"]["meta_dir"]) / f"fusion_meta_clean_train{meta_suffix}.csv"
    train_df    = pd.read_csv(train_csv)

    meta_encoder_path = Path(base_config["data"]["meta_encoder_path"])
    if meta_encoder_path.exists():
        print(f"Loading MetaEncoder from {meta_encoder_path}")
        meta_encoder = MetaEncoder.load(str(meta_encoder_path))
    else:
        print("Fitting MetaEncoder...")
        meta_encoder = MetaEncoder().fit(train_df)
        meta_encoder.save(str(meta_encoder_path))

    summary = {}
    targets = base_config.get("targets", ["popularity", "meanScore"])

    for run_id, dropout, weight_decay, batch_size in SEARCH_SPACE:
        print(f"\n{'='*65}")
        print(f"Run {run_id} | dropout={dropout}  wd={weight_decay:.0e}  batch={batch_size}")
        print(f"{'='*65}")

        cfg = copy.deepcopy(base_config)
        cfg["model"]["dropout"]         = dropout
        cfg["model"]["attn_dropout"]    = round(dropout * 0.4, 3)  # attn_dropout ∝ dropout
        cfg["training"]["weight_decay"] = weight_decay
        cfg["training"]["batch_size"]   = batch_size
        cfg["output"]["run_id"]         = run_id
        cfg["output"]["notes"]          = _make_run_notes(run_id, dropout, weight_decay, batch_size)

        run_result = {
            "run_id":       run_id,
            "dropout":      dropout,
            "weight_decay": weight_decay,
            "batch_size":   batch_size,
            "notes":        cfg["output"]["notes"],
        }

        for target in targets:
            best_val = train_target(target, cfg, meta_encoder)
            # 讀 final_metrics.json 取 val metrics
            metrics_path = (Path(cfg["output"]["run_dir"])
                            / run_id / target / "final_metrics.json")
            with open(metrics_path) as f:
                m = json.load(f)
            run_result[target] = m.get("val", {})
            run_result[target]["best_val_loss"] = round(best_val, 6)

        summary[run_id] = run_result
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nRun {run_id} saved → {SUMMARY_PATH}")

    # ── 最終排名 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Hyperparameter Search Summary")
    print(f"{'='*65}")

    for target in targets:
        print(f"\n--- {target} (val) ---")
        rows = []
        for run_id, r in summary.items():
            vm = r.get(target, {})
            rows.append({
                "run_id":       run_id,
                "dropout":      r["dropout"],
                "wd":           r["weight_decay"],
                "batch":        r["batch_size"],
                "spearman":     vm.get("spearman_rho", float("nan")),
                "best_val_loss":vm.get("best_val_loss", float("nan")),
                **({"log_MAE": vm.get("log_MAE", float("nan"))}
                   if target == "popularity" else
                   {"MAE": vm.get("MAE", float("nan"))}),
            })

        sort_key = "log_MAE" if target == "popularity" else "MAE"
        rows.sort(key=lambda x: x.get(sort_key, float("inf")))
        for rank, row in enumerate(rows, 1):
            extra = f"log_MAE={row.get('log_MAE', '—')}" if target == "popularity" else f"MAE={row.get('MAE', '—')}"
            print(f"  #{rank}  Run{row['run_id']}  drop={row['dropout']} wd={row['wd']:.0e} "
                  f"bs={row['batch']}  spearman={row['spearman']}  {extra}")

    print(f"\nFull results → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
