"""
Seed robustness：Run23/24/25 = Run22 設定（full model + per-target HP），只改 random seed。

  Run22 = seed 42（既有，runs/22）
  Run23 = seed 43
  Run24 = seed 44
  Run25 = seed 45

per-target HP overrides 由 train_target / evaluate 自動套用（與 Run22 完全一致），
唯一變因是 seed → 用來回應老師「single fixed seed 不足」，補 multi-seed mean/std。

結束時自動讀 Run22 + Run23/24/25 共 4 個 seed，輸出各 target 主指標的 mean±std。
摘要：src_2/runs/rerun_seeds_run23_25_summary.json
用法：python src_2/rerun_seeds_run23_25.py
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fussion_training"))

import numpy as np
import yaml

from meta_encoder import MetaEncoder
from train import train_target
from evaluate import evaluate

CONFIG_PATH  = "src_2/fussion_configs.yaml"
RUN_DIR      = Path("src_2/runs")
SUMMARY_PATH = RUN_DIR / "rerun_seeds_run23_25_summary.json"
TARGETS      = ["popularity", "meanScore"]

# (run_id, seed)；Run22(seed=42) 已存在，這裡只補 23/24/25
RUNS = [("23", 43), ("24", 44), ("25", 45)]

# 各 target 報告的主指標（與論文 4.3 一致）
POP_KEYS = ["log_MAE", "log_R2", "spearman_rho", "factor_acc_2x"]
MS_KEYS  = ["MAE", "R2", "spearman_rho", "acc_within_10pt"]


def _load_test_metrics(run_id: str, target: str) -> dict:
    p = RUN_DIR / run_id / target / "final_metrics.json"
    if not p.exists():
        return {}
    return json.load(open(p)).get("test", {})


def main():
    base = yaml.safe_load(open(CONFIG_PATH))
    assert base["output"]["run_id"] == "22", "base config 應為 Run22（run_id=22）"

    meta_encoder = MetaEncoder.load(base["data"]["meta_encoder_path"])
    summary = json.load(open(SUMMARY_PATH)) if SUMMARY_PATH.exists() else {}

    for run_id, seed in RUNS:
        print(f"\n{'='*70}\nRun{run_id} | full model, per-target HP, seed={seed}\n{'='*70}")
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["output"]["run_id"] = run_id
        cfg["output"]["notes"]  = (f"Run{run_id}: = Run22 full model, per-target HP, "
                                   f"seed={seed}（seed robustness）")

        rec = {"run_id": run_id, "seed": seed}
        for target in TARGETS:
            train_target(target, cfg, meta_encoder)
            evaluate(target, "test", cfg, meta_encoder)
            rec[target] = {"test": _load_test_metrics(run_id, target)}
        summary[run_id] = rec
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  saved → {SUMMARY_PATH}")

    # ── mean ± std over 4 seeds: Run22(42) + Run23(43) + Run24(44) + Run25(45) ──
    seed_runs = {"22": 42, "23": 43, "24": 44, "25": 45}
    print(f"\n{'='*70}\nSeed robustness (test set) — mean ± std over seeds 42/43/44/45\n{'='*70}")
    agg = {}
    for target, keys in [("popularity", POP_KEYS), ("meanScore", MS_KEYS)]:
        print(f"\n[{target}]")
        agg[target] = {}
        for k in keys:
            vals = []
            for rid in seed_runs:
                m = _load_test_metrics(rid, target)
                if k in m and m[k] is not None:
                    vals.append(float(m[k]))
            if vals:
                mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                agg[target][k] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(vals)}
                print(f"  {k:16s}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")
            else:
                print(f"  {k:16s}: (no data)")

    summary["_seed_robustness"] = {"seeds": seed_runs, "agg": agg}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✅ done → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
