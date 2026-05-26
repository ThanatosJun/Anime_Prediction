"""
Step 1: Detect anime characters via YOLO and save cropped regions to disk.

Input:
  data/fussion/fusion_meta_clean_{split}{meta_suffix}.csv
  data/image/{split_dir}/{id}_coverImage_extraLarge.jpg

Output:
  data/image/crops/{split}/{id}_crop_{i}.jpg
  data/image/crops/crop_manifest_{split}.parquet
    columns: id (int), crop_path (str)   — one row per crop

Usage:
  conda activate animeprediction
  python -m src.fussion_branch.run_yolo_crop
  python -m src.fussion_branch.run_yolo_crop --splits train val
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from src.fussion_branch.image_components.yolo_detector import detect_and_crop

_DEFAULT_CONFIG = "src/fussion_branch/configs/fusion_config.yaml"

IMAGE_BASE_DIR = Path("data/image")
CROPS_BASE_DIR = Path("data/image/crops")

SPLIT_IMAGE_DIR = {
    "train":           "train_image",
    "val":             "validation_image",
    "test":            "test_image",
    "holdout_unknown": "holdout_unknow_image",
}


def run(
    splits: tuple = ("train", "val", "test", "holdout_unknown"),
    config_path: str = _DEFAULT_CONFIG,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    meta_dir    = Path(cfg["data"]["fusion_meta_dir"])
    meta_suffix = cfg["data"].get("meta_suffix", "")
    yolo_cfg    = cfg.get("yolo", {})

    CROPS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        meta_path = meta_dir / f"fusion_meta_clean_{split}{meta_suffix}.csv"
        if not meta_path.exists():
            print(f"[{split}] fusion_meta_clean not found — skipping")
            continue

        ids = pd.read_csv(meta_path, usecols=["id"])["id"].astype(int).tolist()
        img_dir  = IMAGE_BASE_DIR / SPLIT_IMAGE_DIR[split]
        crop_dir = CROPS_BASE_DIR / split
        crop_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        missing = 0

        for anime_id in tqdm(ids, desc=f"[{split}] YOLO"):
            src = img_dir / f"{anime_id}_coverImage_extraLarge.jpg"
            if not src.exists():
                missing += 1
                continue

            try:
                img = Image.open(src).convert("RGB")
            except Exception:
                missing += 1
                continue

            crops = detect_and_crop(img, **yolo_cfg)

            for i, crop in enumerate(crops):
                out_path = crop_dir / f"{anime_id}_crop_{i}.jpg"
                crop.save(out_path, format="JPEG", quality=95)
                rows.append({"id": anime_id, "crop_path": str(out_path)})

        manifest_path = CROPS_BASE_DIR / f"crop_manifest_{split}.parquet"
        pd.DataFrame(rows).to_parquet(manifest_path, index=False)
        print(f"[{split}] {len(rows)} crops saved  |  missing images: {missing}  → {manifest_path}")

    print("\nYOLO crop generation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO crop: detect anime characters and save crops")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "holdout_unknown"])
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    args = parser.parse_args()
    run(splits=tuple(args.splits), config_path=args.config)


if __name__ == "__main__":
    main()
