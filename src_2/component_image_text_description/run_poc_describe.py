"""
VLM 描述 POC（小規模驗證）

目的：對少量動畫 cover 生成 ToriiGate 文字描述，人工檢查語意品質，
      判斷是否補足 Swin embedding 漏掉的語意（類型/畫風/氛圍/角色關係），
      再決定要不要全量 19k。

用法：
    python src_2/component_image_text_description/run_poc_describe.py            # train 50 部, mode=short
    python src_2/component_image_text_description/run_poc_describe.py --n 30 --mode long
    python src_2/component_image_text_description/run_poc_describe.py --split val --n 20

輸出：src_2/component_image_text_description/poc_descriptions.csv（id, title, has_cover, description）
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))

import pandas as pd
from tqdm import tqdm

from describer import ToriiGateDescriber

HF_MODEL_DIR = _HERE / "model-torii-hf"

# split → 圖片目錄（對齊 run_swin_embedding.py）
_SPLIT_IMAGE_DIR = {
    "train": "train_image",
    "val":   "validation_image",
    "test":  "test_image",
}


def _title(row) -> str:
    for col in ("title_romaji", "title_english"):
        v = getattr(row, col, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(int(row.id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n",     type=int, default=50, help="抽樣動畫數")
    parser.add_argument("--mode",  default="short",
                        help="caption 模式（short 最省時，long_thoughts_v2 最詳細）")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--out",   default=str(_HERE / "poc_descriptions.csv"))
    args = parser.parse_args()

    if not HF_MODEL_DIR.exists() or not (HF_MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError(
            f"找不到 HF 權重：{HF_MODEL_DIR}/model.safetensors\n"
            f"請先下載 Minthy/ToriiGate-0.5 的 model.safetensors 放到該目錄。")

    # ── 抽樣 ──────────────────────────────────────────────────────────────────
    meta_path = _ROOT / "src_2" / "data" / "dataset" / f"fusion_meta_clean_{args.split}_v2.csv"
    df = pd.read_csv(meta_path)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    img_dir = _ROOT / "src_2" / "data" / "image" / _SPLIT_IMAGE_DIR[args.split]

    print(f"Loading ToriiGate from {HF_MODEL_DIR} ...")
    describer = ToriiGateDescriber(model_path=str(HF_MODEL_DIR))

    rows = []
    for row in tqdm(sample.itertuples(index=False), total=len(sample), desc="describe", ncols=90):
        aid    = int(row.id)
        cover  = img_dir / f"{aid}_coverImage_extraLarge.jpg"
        title  = _title(row)
        if not cover.exists():
            rows.append({"id": aid, "title": title, "has_cover": 0, "description": ""})
            continue
        try:
            desc = describer.describe(str(cover), mode=args.mode)
        except Exception as e:
            desc = f"[ERROR] {e}"
        rows.append({"id": aid, "title": title, "has_cover": 1, "description": desc})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    n_ok = int((out_df["has_cover"] == 1).sum())
    print(f"\nSaved {len(out_df)} rows ({n_ok} with cover) → {args.out}")
    # 印前 3 筆供快速檢視
    for r in out_df[out_df["has_cover"] == 1].head(3).itertuples():
        print(f"\n── [{r.id}] {r.title} ──\n{r.description[:400]}")


if __name__ == "__main__":
    main()
