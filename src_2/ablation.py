"""
FusionModel v2 Ablation Study（order.md Step 11 + 12）

baseline 超參（繼承 hp_search 最佳 Run07）：
  dropout=0.3, attn_dropout=0.12, weight_decay=1e-3, batch_size=512,
  TrendHead enabled (pop+score), lr=0.001

對照組（full model）：Run07（use_rag=true, image_mode=cover_banner_yolo）— 已訓練，直接讀取

消融組：
  Step 11  abl_rag_off            use_rag=false                  （移除 Cross Attention）
  Step 12  abl_img_cover          image_mode=cover               （只用封面）
  Step 12  abl_img_cover_banner   image_mode=cover_banner        （封面 + banner）

每組都訓練 popularity + meanScore，並在 test set 評估。
結果摘要：src_2/runs/ablation_summary.json
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
from evaluate import evaluate

CONFIG_PATH  = "src_2/fussion_configs.yaml"
SUMMARY_PATH = Path("src_2/runs/ablation_summary.json")

# baseline 超參（= Run07）
BASE_HP = {
    "dropout":      0.3,
    "attn_dropout": 0.12,
    "weight_decay": 1e-3,
    "batch_size":   512,
}

# 每筆：(run_id, use_rag, image_mode, notes)
ABLATIONS = [
    ("abl_rag_off",          False, "cover_banner_yolo",
     "Step11 RAG off: use_rag=false（移除 Cross Attention，純 MLP）"),
    ("abl_img_cover",        True,  "cover",
     "Step12 image=cover only（移除 banner + yolo）"),
    ("abl_img_cover_banner", True,  "cover_banner",
     "Step12 image=cover+banner（移除 yolo crop）"),
]

TARGETS = ["popularity", "meanScore"]


def _apply_base_hp(cfg):
    cfg["model"]["dropout"]         = BASE_HP["dropout"]
    cfg["model"]["attn_dropout"]    = BASE_HP["attn_dropout"]
    cfg["training"]["weight_decay"] = BASE_HP["weight_decay"]
    cfg["training"]["batch_size"]   = BASE_HP["batch_size"]


def main():
    with open(CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    # MetaEncoder（所有 run 共用）
    meta_encoder_path = Path(base_config["data"]["meta_encoder_path"])
    if not meta_encoder_path.exists():
        raise FileNotFoundError(f"MetaEncoder not found: {meta_encoder_path}. Run train.py first.")
    meta_encoder = MetaEncoder.load(str(meta_encoder_path))

    summary = {}

    for run_id, use_rag, image_mode, notes in ABLATIONS:
        print(f"\n{'='*65}\n{run_id} | use_rag={use_rag} image_mode={image_mode}\n{'='*65}")

        cfg = copy.deepcopy(base_config)
        _apply_base_hp(cfg)
        cfg["use_rag"]          = use_rag
        cfg["image_mode"]       = image_mode
        cfg["output"]["run_id"] = run_id
        cfg["output"]["notes"]  = notes

        run_result = {"run_id": run_id, "use_rag": use_rag,
                      "image_mode": image_mode, "notes": notes}

        for target in TARGETS:
            train_target(target, cfg, meta_encoder)
            evaluate(target, "test", cfg, meta_encoder)
            metrics_path = Path(cfg["output"]["run_dir"]) / run_id / target / "final_metrics.json"
            m = json.load(open(metrics_path))
            run_result[target] = {"val": m.get("val", {}), "test": m.get("test", {})}

        summary[run_id] = run_result
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n{run_id} saved → {SUMMARY_PATH}")

    # ── 對照表（vs Run07 full model）─────────────────────────────────────────
    print(f"\n{'='*65}\nAblation Summary (test set)\n{'='*65}")
    for run_id, r in summary.items():
        print(f"\n{run_id}:")
        for target in TARGETS:
            t = r[target]["test"]
            key = "log_MAE" if target == "popularity" else "MAE"
            print(f"  {target:11s}: spearman={t.get('spearman_rho','—')}  {key}={t.get(key,'—')}")

    print(f"\nFull results → {SUMMARY_PATH}")
    print("對照組（full model）= Run07：src_2/runs/07/{target}/final_metrics.json")


if __name__ == "__main__":
    main()
