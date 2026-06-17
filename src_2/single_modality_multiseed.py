"""
單模態多 seed（Table 9/10 的 w/ metadata、w/ text、w/ image 三列補 7 seed）。
與 ablation_multiseed_t1b.py 同源：Run22 base + per-target HP，唯一變因 = 保留的模態 + seed。

與舊 ablation_multimodal.py 一致的設定：single-modality 一律 rag off + TrendHead off
（避免 release_year 經 trend 洩漏 meta，量的是「單一模態各自上限」）。
注意：此 delta 同時包含「移除其他模態 + rag + trend」，屬 ceiling 量度，非單組件淨效果。

seed：42/43/44/45/247135/610172/796445；full 對應 Run22–28（與 t1b 相同）。
可續跑（已完成 run_id 跳過）。
摘要：src_2/runs/single_modality_multiseed_summary.json（ablated mean±std、delta-vs-full mean±std、paired t-test / Wilcoxon）
用法：python src_2/single_modality_multiseed.py
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fussion_training"))

import numpy as np
import yaml
from scipy import stats

from meta_encoder import MetaEncoder
from train import train_target
from evaluate import evaluate

CONFIG_PATH  = "src_2/fussion_configs.yaml"
RUN_DIR      = Path("src_2/runs")
SUMMARY_PATH = RUN_DIR / "single_modality_multiseed_summary.json"
TARGETS      = ["popularity", "meanScore"]

SEED_TO_FULL = {42: "22", 43: "23", 44: "24", 45: "25",
                247135: "26", 610172: "27", 796445: "28"}
SEEDS = list(SEED_TO_FULL.keys())

# (abl_key, modalities, use_rag, trend_on, notes) — 三組單模態，皆 rag off + trend off
ABLATIONS = [
    ("onlymeta",  {"image": False, "text": False, "meta": True},  False, False, "meta-only（無 rag / trend）"),
    ("onlytext",  {"image": False, "text": True,  "meta": False}, False, False, "text-only（無 rag / trend）"),
    ("onlyimage", {"image": True,  "text": False, "meta": False}, False, False, "image-only cover+banner+yolo（無 rag / trend）"),
]

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
    for abl_key, mods, use_rag, trend_on, notes in ABLATIONS:
        for seed in SEEDS:
            done += 1
            run_id = f"sm_{abl_key}_{seed}"
            if _done(run_id):
                print(f"[{done}/{total}] {run_id} 已完成，跳過")
                continue
            print(f"\n{'='*70}\n[{done}/{total}] {run_id} | {notes} | seed={seed}\n{'='*70}")
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["modalities"] = mods
            cfg["use_rag"] = use_rag
            cfg["model"]["trend_head"] = {
                "enabled":  trend_on,
                "apply_to": ["popularity", "meanScore"] if trend_on else [],
            }
            cfg["output"]["run_id"] = run_id
            cfg["output"]["notes"]  = f"single-modality {abl_key} | {notes} | seed={seed}"
            rec = {"run_id": run_id, "abl": abl_key, "seed": seed}
            for target in TARGETS:
                train_target(target, cfg, meta_encoder)
                evaluate(target, "test", cfg, meta_encoder)
                rec[target] = {"test": _test_metrics(run_id, target)}
            summary[run_id] = rec
            SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(f"  saved → {SUMMARY_PATH}")

    # ── 聚合：ablated mean±std、delta(ablated − full) mean±std、paired t-test / Wilcoxon ──
    print(f"\n{'='*70}\nSingle-modality summary (test) — ablated mean±std & delta-vs-full\n{'='*70}")
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
                    a = _test_metrics(f"sm_{abl_key}_{seed}", target).get(k)
                    f = _test_metrics(SEED_TO_FULL[seed], target).get(k)
                    if a is not None:
                        abl_vals.append(float(a))
                    if a is not None and f is not None:
                        deltas.append(float(a) - float(f))     # ablated − full
                if not abl_vals:
                    print(f"    {k:16s}: （無資料）")
                    continue
                am  = float(np.mean(abl_vals))
                asd = float(np.std(abl_vals, ddof=1)) if len(abl_vals) > 1 else 0.0
                d = np.array(deltas, dtype=float)
                dm = float(d.mean()) if len(d) else None
                dsd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
                p_t = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 else float("nan")
                try:
                    p_w = float(stats.wilcoxon(d).pvalue) if len(d) > 1 else float("nan")
                except Exception:
                    p_w = float("nan")
                agg[abl_key][target][k] = {
                    "ablated_mean": round(am, 4), "ablated_std": round(asd, 4),
                    "delta_mean": round(dm, 4) if dm is not None else None,
                    "delta_std": round(dsd, 4),
                    "p_ttest": round(p_t, 4), "p_wilcoxon": round(p_w, 4),
                    "n": len(d),
                }
                dtxt = f"Δ {dm:+.4f}±{dsd:.4f} p={p_t:.4f}" if dm is not None else "Δ —"
                print(f"    {k:16s}: ablated {am:.4f}±{asd:.4f} | {dtxt} (n={len(d)})")
    summary["_sm_agg"] = {"seeds": SEEDS, "seed_to_full": SEED_TO_FULL, "agg": agg}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✅ done → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
