"""
Per-target HP 版 Ablation（seed=42）—— 供論文 4.3 使用

與舊 ablation.py / ablation_multimodal.py 的差異：
  舊版強制覆寫 BASE_HP（= Run07 共用 HP），meanScore 因此不在最佳 HP。
  本版「不覆寫 HP」，保留 config 的 per-target overrides，
  train_target / evaluate 會各自套用 apply_target_overrides，
  → 每個 target 用它自己的最佳超參數（popularity dr=0.3；meanScore dr=0.3,attn_dr=0.1,wd=1e-4,batch=256）。

對照組（full model）也由本腳本重跑（abl_full_pt = Run22 設定），確保整張表同一條 code path、可比。

消融組（每組重訓，concat_dim 隨啟用分支改變）：
  abl_full_pt        full（image+text+meta+rag）TrendHead on   → 對照組
  abl_rag_off_pt     RAG off（移除 cross-attention，純 MLP）   TrendHead on
  abl_no_image_pt    image 分支移除（text+meta+rag）           TrendHead on
  abl_only_text_pt   只留 text                                 TrendHead off
  abl_only_image_pt  只留 image（cover+banner+yolo）           TrendHead off
  abl_only_meta_pt   只留 meta                                 TrendHead off
  ── 以下為 image 來源消融（4.3.2 細項，可選；不需要可註解掉）──
  abl_img_cover_pt        image=cover only
  abl_img_cover_banner_pt image=cover+banner
  abl_img_banner_pt       image=banner only
  abl_img_yolo_pt         image=yolo only

注意：single-modality 組關閉 TrendHead（避免 release_year 經 trend 洩漏 meta 資訊），
      故與 full 非完全對等，回答的是「單一模態各自上限」。

結果摘要：src_2/runs/ablation_pertarget_s42_summary.json
用法：python src_2/ablation_pertarget_s42.py
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
SUMMARY_PATH = Path("src_2/runs/ablation_pertarget_s42_summary.json")
TARGETS      = ["popularity", "meanScore"]

ALL_MODS = {"image": True, "text": True, "meta": True}

# (run_id, modalities, use_rag, image_mode, trend_on, notes)
ABLATIONS = [
    # ── 對照組 ──────────────────────────────────────────────────────────────
    ("abl_full_pt", ALL_MODS, True, "cover_banner_yolo", True,
     "full model（image+text+meta+rag），per-target HP，control（= Run22）"),

    # ── Temporal trend on/off（concept drift 對照）─────────────────────────
    ("abl_full_notrend_pt", ALL_MODS, True, "cover_banner_yolo", False,
     "full model 但 TrendHead off（對照 abl_full_pt，驗證 linear+year 時序項對 concept drift 的效果）"),

    # ── 4.3.1 With RAG or not ───────────────────────────────────────────────
    ("abl_rag_off_pt", ALL_MODS, False, "cover_banner_yolo", True,
     "RAG off：use_rag=false（移除 cross-attention，純 MLP）"),

    # ── 4.3.2 With modality or not ──────────────────────────────────────────
    ("abl_no_image_pt", {"image": False, "text": True, "meta": True}, True, "cover_banner_yolo", True,
     "no image（text+meta+rag），TrendHead on"),
    ("abl_only_text_pt", {"image": False, "text": True, "meta": False}, False, "cover_banner_yolo", False,
     "text only，無 rag / TrendHead"),
    ("abl_only_image_pt", {"image": True, "text": False, "meta": False}, False, "cover_banner_yolo", False,
     "image only（cover+banner+yolo），無 rag / TrendHead"),
    ("abl_only_meta_pt", {"image": False, "text": False, "meta": True}, False, "cover_banner_yolo", False,
     "meta only，無 rag / TrendHead"),

    # ── image 來源消融（4.3.2 細項，可選）──────────────────────────────────
    ("abl_img_cover_pt", ALL_MODS, True, "cover", True,
     "image=cover only（移除 banner + yolo）"),
    ("abl_img_cover_banner_pt", ALL_MODS, True, "cover_banner", True,
     "image=cover+banner（移除 yolo）"),
    ("abl_img_banner_pt", ALL_MODS, True, "banner", True,
     "image=banner only"),
    ("abl_img_yolo_pt", ALL_MODS, True, "yolo", True,
     "image=yolo only（character crop）"),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None,
                        help="只跑指定 run_id（預設跑全部）；例：--only abl_full_notrend_pt")
    args = parser.parse_args()

    base_config = yaml.safe_load(open(CONFIG_PATH))
    assert base_config.get("seed") == 42, "config seed 必須為 42（重現用）"

    meta_encoder_path = Path(base_config["data"]["meta_encoder_path"])
    if not meta_encoder_path.exists():
        raise FileNotFoundError(f"MetaEncoder not found: {meta_encoder_path}. Run train.py first.")
    meta_encoder = MetaEncoder.load(str(meta_encoder_path))

    summary = json.load(open(SUMMARY_PATH)) if SUMMARY_PATH.exists() else {}

    ablations = [a for a in ABLATIONS if args.only is None or a[0] in args.only]

    for run_id, mods, use_rag, image_mode, trend_on, notes in ablations:
        print(f"\n{'='*70}\n{run_id} | mods={mods} rag={use_rag} image={image_mode} trend={trend_on}\n{'='*70}")

        # 重點：不覆寫 HP，保留 config 的 per-target overrides
        #       train_target / evaluate 會各自套用 apply_target_overrides
        cfg = copy.deepcopy(base_config)
        cfg["modalities"]          = mods
        cfg["use_rag"]             = use_rag
        cfg["image_mode"]          = image_mode
        cfg["model"]["trend_head"] = {
            "enabled":  trend_on,
            "apply_to": ["popularity", "meanScore"] if trend_on else [],
        }
        cfg["output"]["run_id"] = run_id
        cfg["output"]["notes"]  = notes

        rec = {"run_id": run_id, "modalities": mods, "use_rag": use_rag,
               "image_mode": image_mode, "trend": trend_on, "notes": notes}

        for target in TARGETS:
            train_target(target, cfg, meta_encoder)
            evaluate(target, "test", cfg, meta_encoder)
            m = json.load(open(Path(cfg["output"]["run_dir"]) / run_id / target / "final_metrics.json"))
            rec[target] = {"val": m.get("val", {}), "test": m.get("test", {})}

        summary[run_id] = rec
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  saved → {SUMMARY_PATH}")

    # ── 對照表（test set，per-target 最佳 HP）─────────────────────────────────
    print(f"\n{'='*70}\nPer-target HP Ablation Summary (test set, seed=42)\n{'='*70}")
    print(f"{'run':24s} | POP log_MAE  POP rho | MS MAE   MS rho")
    print('-'*60)
    for run_id, r in summary.items():
        p = r["popularity"]["test"]
        m = r["meanScore"]["test"]
        print(f"{run_id:24s} | {p.get('log_MAE','—')!s:10s} {p.get('spearman_rho','—')!s:7s} | "
              f"{m.get('MAE','—')!s:7s} {m.get('spearman_rho','—')!s:6s}")

    print(f"\nFull results → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
