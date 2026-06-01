"""
固定 seed=42 重跑所有 scripted 實驗，存成 _s42 後綴（不覆蓋舊結果）。

涵蓋：
  pooler baseline / stage(+LN, -LN) / hp_search(04-09) / ablation(RAG·image) / multimodal(4)
每組 train(pop+score) + test eval。結果摘要：src_2/runs/rerun_s42_summary.json

用法：
  python src_2/rerun_s42.py            # 全部
  python src_2/rerun_s42.py --dry-run  # 只列出計畫，不訓練
"""

import argparse
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

POOLER_CFG = "src_2/fussion_configs.yaml"
STAGE_CFG  = "src_2/fussion_configs_stages.yaml"
SUMMARY    = Path("src_2/runs/rerun_s42_summary.json")
TARGETS    = ["popularity", "meanScore"]

# hp_search 搜尋空間（dropout, wd）
HP = [("04", 0.3, 1e-4), ("05", 0.4, 5e-4), ("06", 0.5, 1e-4),
      ("07", 0.3, 1e-3), ("08", 0.5, 5e-4), ("09", 0.5, 1e-3)]
# ablation BASE_HP（= Run07）
ABL_HP = {"dropout": 0.3, "attn_dropout": 0.12, "weight_decay": 1e-3, "batch_size": 512}
# ablation（use_rag, image_mode）
ABL_RAGIMG = [("abl_rag_off", False, "cover_banner_yolo"),
              ("abl_img_cover", True, "cover"),
              ("abl_img_cover_banner", True, "cover_banner")]
# multimodal（modalities, use_rag, trend）
ABL_MM = [("abl_no_image",  {"image": False, "text": True,  "meta": True},  True,  True),
          ("abl_only_text", {"image": False, "text": True,  "meta": False}, False, False),
          ("abl_only_image",{"image": True,  "text": False, "meta": False}, False, False),
          ("abl_only_meta", {"image": False, "text": False, "meta": True},  False, False)]


def _set_trend(cfg, on):
    cfg["model"]["trend_head"] = {"enabled": on,
                                  "apply_to": ["popularity", "meanScore"] if on else []}


def build_experiments():
    """回傳 [(run_id, config_dict), ...]，run_id 皆含 _s42 後綴。"""
    pooler = yaml.safe_load(open(POOLER_CFG))
    stage  = yaml.safe_load(open(STAGE_CFG))
    exps = []

    # ── pooler baseline（= 現有 fussion_configs.yaml 超參）────────────────────
    c = copy.deepcopy(pooler); exps.append(("pooler_s42", c))

    # ── stage（+LN / -LN）─────────────────────────────────────────────────────
    c = copy.deepcopy(stage); c["model"]["image_stage_norm"] = True
    exps.append(("stage_ln_s42", c))
    c = copy.deepcopy(stage); c["model"]["image_stage_norm"] = False
    exps.append(("stage_noln_s42", c))

    # ── hp_search 04-09（pooler base，改 dropout/wd）──────────────────────────
    for rid, dr, wd in HP:
        c = copy.deepcopy(pooler)
        c["model"]["dropout"]         = dr
        c["model"]["attn_dropout"]    = round(dr * 0.4, 3)
        c["training"]["weight_decay"] = wd
        c["training"]["batch_size"]   = 512
        _set_trend(c, True)
        exps.append((f"{rid}_s42", c))

    # ── ablation（RAG / image）────────────────────────────────────────────────
    for rid, use_rag, img_mode in ABL_RAGIMG:
        c = copy.deepcopy(pooler)
        c["model"]["dropout"]         = ABL_HP["dropout"]
        c["model"]["attn_dropout"]    = ABL_HP["attn_dropout"]
        c["training"]["weight_decay"] = ABL_HP["weight_decay"]
        c["training"]["batch_size"]   = ABL_HP["batch_size"]
        c["use_rag"]    = use_rag
        c["image_mode"] = img_mode
        _set_trend(c, True)
        exps.append((f"{rid}_s42", c))

    # ── multimodal 消融 ───────────────────────────────────────────────────────
    for rid, mods, use_rag, trend in ABL_MM:
        c = copy.deepcopy(pooler)
        c["model"]["dropout"]         = ABL_HP["dropout"]
        c["model"]["attn_dropout"]    = ABL_HP["attn_dropout"]
        c["training"]["weight_decay"] = ABL_HP["weight_decay"]
        c["training"]["batch_size"]   = ABL_HP["batch_size"]
        c["modalities"] = mods
        c["use_rag"]    = use_rag
        _set_trend(c, trend)
        exps.append((f"{rid}_s42", c))

    # 統一 seed + run_id
    for rid, c in exps:
        c["seed"] = 42
        c["output"]["run_id"] = rid
        c["output"]["notes"]  = f"{rid}: seed=42 重跑（變因對齊對照）"
    return exps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exps = build_experiments()
    print(f"共 {len(exps)} 組實驗 × {len(TARGETS)} target = {len(exps)*len(TARGETS)} 次訓練\n")
    for rid, c in exps:
        m = c["model"]
        print(f"  {rid:24s} img_dim={m['image_dim']} stage_proj={m.get('image_stage_projection',False)} "
              f"stage_norm={m.get('image_stage_norm','-')} use_rag={c['use_rag']} "
              f"mode={c['image_mode']} mods={c.get('modalities','full')} dr={m['dropout']} wd={c['training']['weight_decay']} amp={c['training']['mixed_precision']}")
    if args.dry_run:
        print("\n[dry-run] 不訓練")
        return

    base = yaml.safe_load(open(POOLER_CFG))
    meta_encoder = MetaEncoder.load(base["data"]["meta_encoder_path"])

    summary = {}
    for i, (rid, cfg) in enumerate(exps, 1):
        print(f"\n{'='*65}\n[{i}/{len(exps)}] {rid}\n{'='*65}")
        rec = {"run_id": rid}
        for target in TARGETS:
            train_target(target, cfg, meta_encoder)
            evaluate(target, "test", cfg, meta_encoder)
            m = json.load(open(Path(cfg["output"]["run_dir"]) / rid / target / "final_metrics.json"))
            rec[target] = {"val": m.get("val", {}), "test": m.get("test", {})}
        summary[rid] = rec
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  saved → {SUMMARY}")

    print(f"\n✅ 全部完成 → {SUMMARY}")


if __name__ == "__main__":
    main()
