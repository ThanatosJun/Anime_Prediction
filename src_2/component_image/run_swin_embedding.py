"""
Swin-B image embedding 生成（3 種類型合併為一個 parquet）

輸出：src_2/embedding/image/image_embeddings_{split}.parquet
欄位：id,
      yolo_0  … yolo_1023,  has_yolo,
      cover_0 … cover_1023, has_cover,
      banner_0… banner_1023, has_banner

yolo 需先執行 run_yolo_crop.py。

用法：
    python run_swin_embedding.py                         # train/val/test 全部
    python run_swin_embedding.py --splits train val
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))

from model import load_model, get_embedding
from image_process import load_image, ResizeWithPad, get_transform_original
from config import load_config

_CFG_PATH = str(_HERE / "image_encoder_config.yaml")

_META_DIR  = _ROOT / "src_2" / "data" / "dataset"
_IMAGE_DIR = _ROOT / "src_2" / "data" / "image"
_CROP_DIR  = _ROOT / "src_2" / "data" / "image" / "crops"
_OUT_DIR   = _ROOT / "src_2" / "embedding" / "image"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_EMB_DIM = 1024

_SPLIT_IMAGE_DIR = {
    "train":           "train_image",
    "val":             "validation_image",
    "test":            "test_image",
    "holdout_unknown": "holdout_unknow_image",
}


def _load_swin(config: dict, device: torch.device):
    model_path = config["inference"]["model_path"]
    p = Path(model_path)
    if not p.is_absolute() and not p.exists():
        p = _ROOT / model_path
    return load_model({"model": {"name": str(p)}}).to(device).eval()


def _embed(model, imgs, resize, transform, device) -> np.ndarray:
    batch = torch.stack([transform(resize(i)) for i in imgs]).to(device)
    with torch.no_grad():
        return get_embedding(model, batch).mean(dim=0).cpu().numpy()


def _zero() -> list:
    return [0.0] * _EMB_DIM


def run_swin_embedding(splits: list):
    config = load_config(_CFG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_swin(config, device)

    image_size = config["data"]["image_size"]
    resize     = ResizeWithPad(image_size)
    transform  = get_transform_original(image_size)

    yolo_cols   = [f"yolo_{i}"   for i in range(_EMB_DIM)]
    cover_cols  = [f"cover_{i}"  for i in range(_EMB_DIM)]
    banner_cols = [f"banner_{i}" for i in range(_EMB_DIM)]

    for split in splits:
        csv_path = _META_DIR / f"fusion_meta_clean_{split}_v2.csv"
        if not csv_path.exists():
            print(f"[{split}] CSV not found — skip")
            continue

        img_dir  = _IMAGE_DIR / _SPLIT_IMAGE_DIR.get(split, f"{split}_image")
        crop_dir = _CROP_DIR / split
        df       = pd.read_csv(csv_path)
        print(f"\n[{split}] {len(df)} rows")

        rows = []
        stats = {"no_yolo": 0, "no_cover": 0, "no_banner": 0}

        for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
            anime_id = int(row["id"])
            r = {"id": anime_id}

            # ── yolo ──────────────────────────────────────────────────────────
            crop_files = sorted(crop_dir.glob(f"{anime_id}_crop_*.jpg")) if crop_dir.exists() else []
            imgs       = [load_image(str(f)) for f in crop_files]
            imgs       = [i for i in imgs if i is not None]
            if imgs:
                r.update(dict(zip(yolo_cols, _embed(model, imgs, resize, transform, device).tolist())))
                r["has_yolo"] = 1
            else:
                r.update(dict(zip(yolo_cols, _zero())))
                r["has_yolo"] = 0
                stats["no_yolo"] += 1

            # ── cover ─────────────────────────────────────────────────────────
            cover_file = img_dir / f"{anime_id}_coverImage_extraLarge.jpg"
            img = load_image(str(cover_file)) if cover_file.exists() else None
            if img:
                r.update(dict(zip(cover_cols, _embed(model, [img], resize, transform, device).tolist())))
                r["has_cover"] = 1
            else:
                r.update(dict(zip(cover_cols, _zero())))
                r["has_cover"] = 0
                stats["no_cover"] += 1

            # ── banner ────────────────────────────────────────────────────────
            banner_file = img_dir / f"{anime_id}_bannerImage.jpg"
            img = load_image(str(banner_file)) if banner_file.exists() else None
            if img:
                r.update(dict(zip(banner_cols, _embed(model, [img], resize, transform, device).tolist())))
                r["has_banner"] = 1
            else:
                r.update(dict(zip(banner_cols, _zero())))
                r["has_banner"] = 0
                stats["no_banner"] += 1

            rows.append(r)

        out_df   = pd.DataFrame(rows)
        out_path = _OUT_DIR / f"image_embeddings_{split}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(out_df)} rows → {out_path.relative_to(_ROOT)}")
        print(f"  no_yolo={stats['no_yolo']}  no_cover={stats['no_cover']}  no_banner={stats['no_banner']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    run_swin_embedding(args.splits)


if __name__ == "__main__":
    main()
