"""
補跑 seed=42 實驗，append 進 rerun_s42_summary.json：
  02_s42            Run02 原設定（dropout=0.3, wd=1e-4, batch=256, trend on）→ 驗證舊 meanScore 7.29 是否 seed 運氣
  03_s42            Run03 原設定（= base config，應與 pooler_s42 相同）
  abl_img_banner_s42  單模態：只有 banner
  abl_img_yolo_s42    單模態：只有 character（yolo crop）
（cover-only 已有 abl_img_cover_s42，三者可做 cover/banner/yolo 對照）

用法：python src_2/rerun_extra_s42.py
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

POOLER_CFG = "src_2/fussion_configs.yaml"
SUMMARY    = Path("src_2/runs/rerun_s42_summary.json")
TARGETS    = ["popularity", "meanScore"]

# 單模態 image 用的 baseline hp（= ablation BASE_HP = Run07）
ABL_HP = {"dropout": 0.3, "attn_dropout": 0.12, "weight_decay": 1e-3, "batch_size": 512}


def _set_trend(c, on):
    c["model"]["trend_head"] = {"enabled": on,
                                "apply_to": ["popularity", "meanScore"] if on else []}


def build():
    pooler = yaml.safe_load(open(POOLER_CFG))
    exps = []

    # Run02：dropout=0.3, attn_dropout=0.1, wd=1e-4, batch=256, trend on, temporal_w on
    c = copy.deepcopy(pooler)
    c["model"]["dropout"]         = 0.3
    c["model"]["attn_dropout"]    = 0.1
    c["training"]["weight_decay"] = 1e-4
    c["training"]["batch_size"]   = 256
    _set_trend(c, True)
    exps.append(("02_s42", c, "Run02 原設定 seed=42（dr=0.3, wd=1e-4, batch=256）"))

    # Run03：= base config（dropout=0.5, attn_drop=0.2, wd=1e-3, batch=512）
    c = copy.deepcopy(pooler)
    exps.append(("03_s42", c, "Run03 原設定 seed=42（= base config，應同 pooler_s42）"))

    # 單模態 image：banner only / yolo only
    for rid, mode in [("abl_img_banner_s42", "banner"), ("abl_img_yolo_s42", "yolo")]:
        c = copy.deepcopy(pooler)
        c["model"]["dropout"]         = ABL_HP["dropout"]
        c["model"]["attn_dropout"]    = ABL_HP["attn_dropout"]
        c["training"]["weight_decay"] = ABL_HP["weight_decay"]
        c["training"]["batch_size"]   = ABL_HP["batch_size"]
        c["image_mode"] = mode
        c["use_rag"]    = True
        _set_trend(c, True)
        exps.append((rid, c, f"單模態 image={mode}（對照 cover-only=abl_img_cover_s42）"))

    for rid, c, notes in exps:
        c["seed"] = 42
        c["output"]["run_id"] = rid
        c["output"]["notes"]  = notes
    return exps


def main():
    exps = build()
    print(f"補跑 {len(exps)} 組 × {len(TARGETS)} target\n")
    for rid, c, notes in exps:
        print(f"  {rid:22s} image_mode={c['image_mode']:18s} dr={c['model']['dropout']} "
              f"wd={c['training']['weight_decay']} batch={c['training']['batch_size']}")

    base = yaml.safe_load(open(POOLER_CFG))
    meta_encoder = MetaEncoder.load(base["data"]["meta_encoder_path"])

    summary = json.load(open(SUMMARY)) if SUMMARY.exists() else {}
    for i, (rid, cfg, _) in enumerate(exps, 1):
        print(f"\n{'='*65}\n[{i}/{len(exps)}] {rid}\n{'='*65}")
        rec = {"run_id": rid}
        for target in TARGETS:
            train_target(target, cfg, meta_encoder)
            evaluate(target, "test", cfg, meta_encoder)
            m = json.load(open(Path(cfg["output"]["run_dir"]) / rid / target / "final_metrics.json"))
            rec[target] = {"val": m.get("val", {}), "test": m.get("test", {})}
        summary[rid] = rec
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  saved → {SUMMARY}")

    print(f"\n✅ 補跑完成 → {SUMMARY}")


if __name__ == "__main__":
    main()
