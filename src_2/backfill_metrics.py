"""
Backfill metrics：補齊舊 run 的 final_metrics.json 缺少的指標欄位
（factor_acc_2x / acc_within_10pt 等後加的指標）

不重訓，只重算：
  - test split：直接從 pred_test.csv 重算（純預測函數，不載模型）
  - train/val split：載 checkpoint 重跑推論（複用 evaluate.evaluate 的公式）

各 run 架構從下列來源重建：
  - 01：無 TrendHead；02~09：TrendHead on；全部 full 四分支 / image=cover_banner_yolo
  - ablation：從 ablation_summary.json / ablation_multimodal_summary.json 讀回設定

用法：
    python src_2/backfill_metrics.py            # 全部 run
    python src_2/backfill_metrics.py --dry-run  # 只列出要補什麼，不動檔
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fussion_training"))

import numpy as np
import pandas as pd
import yaml

from meta_encoder import MetaEncoder
from evaluate import evaluate

CONFIG_PATH = "src_2/fussion_configs.yaml"
RUNS_DIR    = Path("src_2/runs")

POP_FIELDS   = {"spearman_rho", "log_R2", "MAE", "log_MAE", "factor_acc_2x"}
SCORE_FIELDS = {"spearman_rho", "R2", "MAE", "acc_within_10pt"}


# ── 各 run 架構設定 ───────────────────────────────────────────────────────────

def _run_overrides(run_id: str) -> dict:
    """回傳該 run 相對 base config 的覆寫（架構相關，dropout 等不影響 eval 故略）。"""
    # hp / 數字 run：full 四分支、image=cover_banner_yolo、use_rag=true
    if run_id in {"01", "02", "03", "04", "05", "06", "07", "08", "09"}:
        trend = run_id != "01"   # Run01 無 TrendHead
        return {"modalities": None, "use_rag": True,
                "image_mode": "cover_banner_yolo", "trend": trend}

    abl = {
        "abl_rag_off":          {"modalities": None, "use_rag": False,
                                 "image_mode": "cover_banner_yolo", "trend": True},
        "abl_img_cover":        {"modalities": None, "use_rag": True,
                                 "image_mode": "cover", "trend": True},
        "abl_img_cover_banner": {"modalities": None, "use_rag": True,
                                 "image_mode": "cover_banner", "trend": True},
        "abl_no_image":         {"modalities": {"image": False, "text": True, "meta": True},
                                 "use_rag": True, "image_mode": "cover_banner_yolo", "trend": True},
        "abl_only_text":        {"modalities": {"image": False, "text": True, "meta": False},
                                 "use_rag": False, "image_mode": "cover_banner_yolo", "trend": False},
        "abl_only_image":       {"modalities": {"image": True, "text": False, "meta": False},
                                 "use_rag": False, "image_mode": "cover_banner_yolo", "trend": False},
        "abl_only_meta":        {"modalities": {"image": False, "text": False, "meta": True},
                                 "use_rag": False, "image_mode": "cover_banner_yolo", "trend": False},
    }
    return abl[run_id]


def _build_config(base: dict, run_id: str) -> dict:
    ov = _run_overrides(run_id)
    cfg = copy.deepcopy(base)
    cfg["output"]["run_id"] = run_id
    cfg["use_rag"]    = ov["use_rag"]
    cfg["image_mode"] = ov["image_mode"]
    if ov["modalities"] is None:
        cfg.pop("modalities", None)
    else:
        cfg["modalities"] = ov["modalities"]
    cfg["model"]["trend_head"] = {
        "enabled":  ov["trend"],
        "apply_to": ["popularity", "meanScore"] if ov["trend"] else [],
    }
    return cfg


# ── test split：從 pred CSV 重算（不載模型）──────────────────────────────────

def _backfill_test_from_csv(run_dir: Path, target: str, metrics: dict) -> bool:
    pred_csv = run_dir / "pred_test.csv"
    if "test" not in metrics or not pred_csv.exists():
        return False
    df = pd.read_csv(pred_csv)
    pred, true = df["pred"].to_numpy(), df["target"].to_numpy()
    m = metrics["test"]
    if target == "popularity":
        lp = np.log1p(np.clip(pred, 0, None)); lt = np.log1p(np.clip(true, 0, None))
        m["factor_acc_2x"] = round(float(np.mean(np.abs(lp - lt) < np.log(2))), 4)
    else:
        m["acc_within_10pt"] = round(float(np.mean(np.abs(pred - true) < 10)), 4)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = yaml.safe_load(open(CONFIG_PATH))
    meta_encoder = MetaEncoder.load(base["data"]["meta_encoder_path"])

    # 找出所有缺欄位的 (run, target, split)
    todo = {}  # (run_id, target) -> {splits needing model recompute}
    for fm in sorted(RUNS_DIR.glob("*/*/final_metrics.json")):
        run_id, target = fm.parts[-3], fm.parts[-2]
        m = json.load(open(fm))
        expected = POP_FIELDS if target == "popularity" else SCORE_FIELDS
        for split in ("train", "val", "test"):
            if split in m and (expected - set(m[split].keys())):
                todo.setdefault((run_id, target), set()).add(split)

    if not todo:
        print("✅ 沒有缺欄位，無需 backfill")
        return

    print(f"需 backfill：{len(todo)} 個 (run, target)\n")
    for (run_id, target), splits in sorted(todo.items()):
        print(f"  {run_id}/{target}: {sorted(splits)}")
    if args.dry_run:
        print("\n[dry-run] 不動檔")
        return

    print()
    for (run_id, target), splits in sorted(todo.items()):
        fm_path = RUNS_DIR / run_id / target / "final_metrics.json"
        metrics = json.load(open(fm_path))
        run_dir = RUNS_DIR / run_id / target

        # 1) test：從 CSV 快速補
        if "test" in splits and _backfill_test_from_csv(run_dir, target, metrics):
            fm_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
            splits.discard("test")
            print(f"  [csv ] {run_id}/{target} test ✓")

        # 2) train/val：載 checkpoint 重跑（evaluate 會 merge 回 final_metrics.json）
        if splits:
            cfg = _build_config(base, run_id)
            for split in sorted(splits):
                evaluate(target, split, cfg, meta_encoder)
                print(f"  [model] {run_id}/{target} {split} ✓")

    print("\n✅ backfill 完成")


if __name__ == "__main__":
    main()
