"""
FusionModel v2 Multimodal Ablation（重訓版，架構真的移除分支）

baseline 超參（= Run07）：dropout=0.3, attn_dropout=0.12, wd=1e-3, batch=512, lr=0.001

消融組（每組重新訓練，concat_dim 隨啟用分支改變）：
  abl_no_image    image 分支整體移除（text + meta + rag）   TrendHead on  → 對照 Run07 full
  abl_only_text   只留 text                                 TrendHead off
  abl_only_image  只留 image（cover+banner+yolo）           TrendHead off
  abl_only_meta   只留 meta                                  TrendHead off

對照組（full model）：Run07 — 已訓練。
結果摘要：src_2/runs/ablation_multimodal_summary.json

注意：single-modality 組關閉 TrendHead（避免 release_year 經 trend 洩漏 meta 資訊），
      故與 Run07 非完全對等，回答的是「單一模態各自上限」。
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fussion_training"))

import yaml

from meta_encoder import MetaEncoder
from train import train_target
from evaluate import evaluate

CONFIG_PATH  = "src_2/fussion_configs.yaml"
SUMMARY_PATH = Path("src_2/runs/ablation_multimodal_summary.json")

BASE_HP = {"dropout": 0.3, "attn_dropout": 0.12, "weight_decay": 1e-3, "batch_size": 512}

# (run_id, modalities, use_rag, trend_on, notes)
ABLATIONS = [
    ("abl_no_image", {"image": False, "text": True, "meta": True}, True, True,
     "multimodal: image 分支移除（text+meta+rag），TrendHead on"),
    ("abl_only_text", {"image": False, "text": True, "meta": False}, False, False,
     "multimodal: text-only，無 rag / TrendHead"),
    ("abl_only_image", {"image": True, "text": False, "meta": False}, False, False,
     "multimodal: image-only（cover+banner+yolo），無 rag / TrendHead"),
    ("abl_only_meta", {"image": False, "text": False, "meta": True}, False, False,
     "multimodal: meta-only，無 rag / TrendHead"),
]

TARGETS = ["popularity", "meanScore"]


def main():
    base_config = yaml.safe_load(open(CONFIG_PATH))

    meta_encoder_path = Path(base_config["data"]["meta_encoder_path"])
    if not meta_encoder_path.exists():
        raise FileNotFoundError(f"MetaEncoder not found: {meta_encoder_path}. Run train.py first.")
    meta_encoder = MetaEncoder.load(str(meta_encoder_path))

    summary = {}

    for run_id, mods, use_rag, trend_on, notes in ABLATIONS:
        print(f"\n{'='*65}\n{run_id} | modalities={mods} rag={use_rag} trend={trend_on}\n{'='*65}")

        cfg = copy.deepcopy(base_config)
        cfg["model"]["dropout"]         = BASE_HP["dropout"]
        cfg["model"]["attn_dropout"]    = BASE_HP["attn_dropout"]
        cfg["training"]["weight_decay"] = BASE_HP["weight_decay"]
        cfg["training"]["batch_size"]   = BASE_HP["batch_size"]
        cfg["modalities"]               = mods
        cfg["use_rag"]                  = use_rag
        cfg["model"]["trend_head"]      = {
            "enabled":  trend_on,
            "apply_to": ["popularity", "meanScore"] if trend_on else [],
        }
        cfg["output"]["run_id"] = run_id
        cfg["output"]["notes"]  = notes

        run_result = {"run_id": run_id, "modalities": mods,
                      "use_rag": use_rag, "trend": trend_on, "notes": notes}

        for target in TARGETS:
            train_target(target, cfg, meta_encoder)
            evaluate(target, "test", cfg, meta_encoder)
            m = json.load(open(Path(cfg["output"]["run_dir"]) / run_id / target / "final_metrics.json"))
            run_result[target] = {"val": m.get("val", {}), "test": m.get("test", {})}

        summary[run_id] = run_result
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n{run_id} saved → {SUMMARY_PATH}")

    # ── 對照表 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nMultimodal Ablation Summary (test set)\n{'='*65}")
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
