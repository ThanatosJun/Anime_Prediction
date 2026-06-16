"""
T1b：Exp2 ablation 多 seed（7 seed 全配對）—— 回應老師 Q3「ablation delta 跨 seed 穩不穩」

3 組關鍵 ablation，各在 7 個 seed 上跑，與 full（Run22–28，同 seed）配對：
  ragoff   : use_rag=False（移除 RAG / cross-attention）
  noimg    : 移除 image 分支（text+meta+rag）
  notrend  : TrendHead off（移除時序項）

seed（與 rerun_seeds.py 一致）：42/43/44/45/247135/610172/796445
full 對應 run：42→22, 43→23, 44→24, 45→25, 247135→26, 610172→27, 796445→28

per-target HP overrides 由 train_target/evaluate 自動套用（與 full 一致），唯一變因 = 被移除的組件 + seed。
可續跑（已完成的 run_id 跳過）。
摘要：src_2/runs/ablation_multiseed_t1b_summary.json（含 ablated mean±std 與 delta-vs-full mean±std）
用法：python src_2/ablation_multiseed_t1b.py
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
SUMMARY_PATH = RUN_DIR / "ablation_multiseed_t1b_summary.json"
TARGETS      = ["popularity", "meanScore"]

ALL_MODS = {"image": True, "text": True, "meta": True}

# seed → full model run_id（Run22–28）
SEED_TO_FULL = {42: "22", 43: "23", 44: "24", 45: "25",
                247135: "26", 610172: "27", 796445: "28"}
SEEDS = list(SEED_TO_FULL.keys())

# (abl_key, modalities, use_rag, image_mode, trend_on, notes)
ABLATIONS = [
    ("ragoff",  ALL_MODS, False, "cover_banner_yolo", True,  "RAG off（移除 cross-attention）"),
    ("noimg",   {"image": False, "text": True, "meta": True}, True, "cover_banner_yolo", True,
     "移除 image 分支（text+meta+rag）"),
    ("notrend", ALL_MODS, True,  "cover_banner_yolo", False, "TrendHead off（移除時序項）"),
]

# 主指標（delta 用）；error 指標越低越好，其餘越高越好
POP_KEYS = ["log_MAE", "log_R2", "spearman_rho", "factor_acc_2x"]
MS_KEYS  = ["MAE", "R2", "spearman_rho", "acc_within_10pt"]
LOWER_BETTER = {"log_MAE", "MAE"}


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

    total = len(ABLATIONS) * len(SEEDS)
    done = 0
    for abl_key, mods, use_rag, image_mode, trend_on, notes in ABLATIONS:
        for seed in SEEDS:
            done += 1
            run_id = f"t1b_{abl_key}_{seed}"
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
            cfg["output"]["notes"]  = f"T1b {abl_key} | {notes} | seed={seed}"
            rec = {"run_id": run_id, "abl": abl_key, "seed": seed}
            for target in TARGETS:
                train_target(target, cfg, meta_encoder)
                evaluate(target, "test", cfg, meta_encoder)
                rec[target] = {"test": _test_metrics(run_id, target)}
            summary[run_id] = rec
            SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"  saved → {SUMMARY_PATH}")

    # ── 聚合：每個 ablation 的 ablated mean±std，以及 delta(ablated − full) mean±std ──
    print(f"\n{'='*70}\nT1b summary (test) — ablated mean±std & delta-vs-full mean±std\n{'='*70}")
    agg = {}
    for abl_key, *_ in ABLATIONS:
        agg[abl_key] = {}
        print(f"\n##### {abl_key} #####")
        for target, keys in [("popularity", POP_KEYS), ("meanScore", MS_KEYS)]:
            agg[abl_key][target] = {}
            print(f"  [{target}]")
            for k in keys:
                abl_vals, deltas = [], []
                for seed in SEEDS:
                    a = _test_metrics(f"t1b_{abl_key}_{seed}", target).get(k)
                    f = _test_metrics(SEED_TO_FULL[seed], target).get(k)
                    if a is not None:
                        abl_vals.append(float(a))
                    if a is not None and f is not None:
                        d = float(a) - float(f)              # ablated − full
                        deltas.append(d)
                if abl_vals:
                    am, asd = np.mean(abl_vals), (np.std(abl_vals, ddof=1) if len(abl_vals) > 1 else 0.0)
                    dm, dsd = (np.mean(deltas), (np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0)) if deltas else (None, None)
                    sign = "↑helps" if ((k in LOWER_BETTER and dm and dm > 0) or
                                        (k not in LOWER_BETTER and dm and dm < 0)) else ""
                    agg[abl_key][target][k] = {
                        "ablated_mean": round(am, 4), "ablated_std": round(asd, 4),
                        "delta_mean": round(dm, 4) if dm is not None else None,
                        "delta_std": round(dsd, 4) if dsd is not None else None,
                        "n": len(deltas),
                    }
                    dtxt = f"Δ {dm:+.4f}±{dsd:.4f}" if dm is not None else "Δ —"
                    print(f"    {k:16s}: ablated {am:.4f}±{asd:.4f} | {dtxt} (n={len(deltas)}) {sign}")
    summary["_t1b_agg"] = {"seeds": SEEDS, "seed_to_full": SEED_TO_FULL, "agg": agg}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✅ done → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
