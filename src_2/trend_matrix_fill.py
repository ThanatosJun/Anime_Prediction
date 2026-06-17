"""
補滿「每個模塊 × with/without trend」矩陣的缺角（× 7 seed）。
與 ablation_multiseed_t1b.py / single_modality_multiseed.py 同源：Run22 base + per-target HP。

已存在（不在此腳本）：
  full          trend ON  = Run22–28          ；trend OFF = t1b_notrend_*
  ragoff        trend ON  = t1b_ragoff_*
  noimg         trend ON  = t1b_noimg_*
  onlymeta/text/image  trend OFF = sm_only*_*

本腳本補（5 組 × 7 seed = 35 次訓練）：
  tm_ragoff_notrend_*     RAG off, trend OFF
  tm_noimg_notrend_*      −image,  trend OFF
  tm_onlymeta_trend_*     meta-only,  trend ON
  tm_onlytext_trend_*     text-only,  trend ON  ⚠️ release_year 經 trend 洩漏（year leak）
  tm_onlyimage_trend_*    image-only, trend ON  ⚠️ year leak

可續跑（已完成 run_id 跳過）。
摘要：src_2/runs/trend_matrix_fill_summary.json
用法：python src_2/trend_matrix_fill.py
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
SUMMARY_PATH = RUN_DIR / "trend_matrix_fill_summary.json"
TARGETS      = ["popularity", "meanScore"]

SEED_TO_FULL = {42: "22", 43: "23", 44: "24", 45: "25",
                247135: "26", 610172: "27", 796445: "28"}
SEEDS = list(SEED_TO_FULL.keys())

ALL_MODS = {"image": True, "text": True, "meta": True}

# (run_prefix, modalities, use_rag, image_mode, trend_on, notes)
FILLS = [
    ("tm_ragoff_notrend",  ALL_MODS, False, "cover_banner_yolo", False, "RAG off, trend OFF"),
    ("tm_noimg_notrend",   {"image": False, "text": True, "meta": True}, True, "cover_banner_yolo", False,
     "-image, trend OFF"),
    ("tm_onlymeta_trend",  {"image": False, "text": False, "meta": True}, False, "cover_banner_yolo", True,
     "meta-only, trend ON"),
    ("tm_onlytext_trend",  {"image": False, "text": True, "meta": False}, False, "cover_banner_yolo", True,
     "text-only, trend ON (year leak)"),
    ("tm_onlyimage_trend", {"image": True, "text": False, "meta": False}, False, "cover_banner_yolo", True,
     "image-only, trend ON (year leak)"),
]

POP_KEYS = ["log_MAE", "log_R2", "spearman_rho", "factor_acc_2x"]
MS_KEYS  = ["MAE", "R2", "spearman_rho", "acc_within_10pt"]


def _test_metrics(run_id: str, target: str) -> dict:
    p = RUN_DIR / run_id / target / "final_metrics.json"
    return json.load(open(p)).get("test", {}) if p.exists() else {}


def _done(run_id: str) -> bool:
    return all(_test_metrics(run_id, t) for t in TARGETS)


def main():
    base = yaml.safe_load(open(CONFIG_PATH))
    assert base["output"]["run_id"] == "22", "base config 應為 Run22"
    meta_encoder = MetaEncoder.load(base["data"]["meta_encoder_path"])
    summary = json.load(open(SUMMARY_PATH)) if SUMMARY_PATH.exists() else {}

    total = len(FILLS) * len(SEEDS)
    done = 0
    for prefix, mods, use_rag, image_mode, trend_on, notes in FILLS:
        for seed in SEEDS:
            done += 1
            run_id = f"{prefix}_{seed}"
            if _done(run_id):
                print(f"[{done}/{total}] {run_id} 已完成，跳過")
                continue
            print(f"\n{'='*70}\n[{done}/{total}] {run_id} | {notes} | seed={seed}\n{'='*70}")
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["modalities"] = mods
            cfg["use_rag"] = use_rag
            cfg["image_mode"] = image_mode
            cfg["model"]["trend_head"] = {
                "enabled":  trend_on,
                "apply_to": ["popularity", "meanScore"] if trend_on else [],
            }
            cfg["output"]["run_id"] = run_id
            cfg["output"]["notes"]  = f"trend-matrix fill | {notes} | seed={seed}"
            rec = {"run_id": run_id, "notes": notes, "seed": seed}
            for target in TARGETS:
                train_target(target, cfg, meta_encoder)
                evaluate(target, "test", cfg, meta_encoder)
                rec[target] = {"test": _test_metrics(run_id, target)}
            summary[run_id] = rec
            SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"  saved → {SUMMARY_PATH}")

    # ── 聚合：每個 fill 的 mean±std ──────────────────────────────────────────────
    print(f"\n{'='*70}\nTrend-matrix fill summary (test) — mean±std over 7 seeds\n{'='*70}")
    agg = {}
    for prefix, *_ in FILLS:
        agg[prefix] = {}
        print(f"\n##### {prefix} #####")
        for target, keys in [("popularity", POP_KEYS), ("meanScore", MS_KEYS)]:
            agg[prefix][target] = {}
            vals_by_k = {k: [] for k in keys}
            for seed in SEEDS:
                t = _test_metrics(f"{prefix}_{seed}", target)
                for k in keys:
                    if t.get(k) is not None:
                        vals_by_k[k].append(float(t[k]))
            line = []
            for k in keys:
                v = vals_by_k[k]
                if v:
                    mean = float(np.mean(v)); std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                    agg[prefix][target][k] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(v)}
                    line.append(f"{k}={mean:.4f}±{std:.4f}")
            print(f"  [{target}] " + "  ".join(line))
    summary["_trend_matrix_agg"] = {"seeds": SEEDS, "agg": agg}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✅ done → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
