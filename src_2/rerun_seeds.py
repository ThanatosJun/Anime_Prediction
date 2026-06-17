"""
Seed robustness：同一 Run22 設定（full model + per-target HP），用多個不同 seed 各跑一次，
報告 mean±std，回應老師「single fixed seed 不足」。

SEED_MAP（run_id → seed）：
  22 → 42       （既有，runs/22）
  23 → 43       （連續）
  24 → 44       （連續）
  25 → 45       （連續）
  26 → 247135   （random，SystemRandom 抽樣後固定記錄）
  27 → 610172   （random）
  28 → 796445   （random）

只訓練 TRAIN_IDS（尚無結果者）；最後對 SEED_MAP 全部有結果的 run 算 mean±std。
摘要：src_2/runs/rerun_seeds_summary.json
用法：python src_2/rerun_seeds.py
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
SUMMARY_PATH = RUN_DIR / "rerun_seeds_summary.json"
TARGETS      = ["popularity", "meanScore"]

# run_id → seed（全部同一 Run22 設定，唯一變因是 seed）
SEED_MAP = {
    "22": 42,
    "23": 43, "24": 44, "25": 45,           # 連續
    "26": 247135, "27": 610172, "28": 796445,  # random（已抽樣固定）
}
# 本次要訓練的（22~25 已存在，不重跑）
TRAIN_IDS = ["26", "27", "28"]

POP_KEYS = ["log_MAE", "log_R2", "spearman_rho", "factor_acc_2x"]
MS_KEYS  = ["MAE", "R2", "spearman_rho", "acc_within_10pt"]


def _load_test_metrics(run_id: str, target: str) -> dict:
    p = RUN_DIR / run_id / target / "final_metrics.json"
    return json.load(open(p)).get("test", {}) if p.exists() else {}


def main():
    base = yaml.safe_load(open(CONFIG_PATH))
    assert base["output"]["run_id"] == "22", "base config 應為 Run22（run_id=22）"
    meta_encoder = MetaEncoder.load(base["data"]["meta_encoder_path"])
    summary = json.load(open(SUMMARY_PATH)) if SUMMARY_PATH.exists() else {}

    for run_id in TRAIN_IDS:
        seed = SEED_MAP[run_id]
        print(f"\n{'='*70}\nRun{run_id} | full model, per-target HP, seed={seed} (random)\n{'='*70}")
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["output"]["run_id"] = run_id
        cfg["output"]["notes"]  = (f"Run{run_id}: = Run22 full model, per-target HP, "
                                   f"seed={seed}（seed robustness, random seed）")
        rec = {"run_id": run_id, "seed": seed}
        for target in TARGETS:
            train_target(target, cfg, meta_encoder)
            evaluate(target, "test", cfg, meta_encoder)
            rec[target] = {"test": _load_test_metrics(run_id, target)}
        summary[run_id] = rec
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  saved → {SUMMARY_PATH}")

    # ── mean ± std over 所有有結果的 seed ────────────────────────────────────
    present = [rid for rid in SEED_MAP
               if _load_test_metrics(rid, "popularity") and _load_test_metrics(rid, "meanScore")]
    print(f"\n{'='*70}\nSeed robustness (test) — mean ± std over {len(present)} seeds: "
          f"{[SEED_MAP[r] for r in present]}\n{'='*70}")
    agg = {}
    for target, keys in [("popularity", POP_KEYS), ("meanScore", MS_KEYS)]:
        print(f"\n[{target}]")
        agg[target] = {}
        for k in keys:
            vals = [float(_load_test_metrics(r, target)[k]) for r in present
                    if _load_test_metrics(r, target).get(k) is not None]
            if vals:
                mean = float(np.mean(vals))
                std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                agg[target][k] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(vals)}
                print(f"  {k:16s}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")
    summary["_seed_robustness"] = {"seed_map": {r: SEED_MAP[r] for r in present}, "agg": agg}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✅ done → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
