"""
根據 fusion_meta_clean_{split}.csv 的 id，
將 data/image/ 內的圖片複製到對應資料夾：
  data/image/train_image/
  data/image/validation_image/
  data/image/test_image/

Usage:
    conda activate animeprediction
    python scripts/split_images_by_split.py
"""
import shutil
from pathlib import Path

import pandas as pd

SRC_DIR = Path("data/image")
IMAGE_TYPES = ["coverImage_medium", "bannerImage"]

SPLITS = {
    "train": {
        "csv":    "data/fussion/fusion_meta_clean_train.csv",
        "dst":    SRC_DIR / "train_image",
    },
    "val": {
        "csv":    "data/fussion/fusion_meta_clean_val.csv",
        "dst":    SRC_DIR / "validation_image",
    },
    "test": {
        "csv":    "data/fussion/fusion_meta_clean_test.csv",
        "dst":    SRC_DIR / "test_image",
    },
    "holdout_unknown": {
        "csv":    "data/fussion/fusion_meta_clean_holdout_unknown.csv",
        "dst":    SRC_DIR / "holdout_unknow_image",
    },
}


def main():
    for split, cfg in SPLITS.items():
        dst = cfg["dst"]
        dst.mkdir(parents=True, exist_ok=True)

        ids = pd.read_csv(cfg["csv"])["id"].tolist()
        found = no_cover = no_banner = 0

        for anime_id in ids:
            for img_type in IMAGE_TYPES:
                src = SRC_DIR / f"{anime_id}_{img_type}.jpg"
                if src.exists():
                    shutil.move(str(src), dst / src.name)
                    found += 1
                elif img_type == "coverImage_medium":
                    no_cover += 1
                else:
                    no_banner += 1

        print(f"[{split:20s}]  ids={len(ids):5d}  moved={found:6d}  no_cover={no_cover:4d}  no_banner={no_banner:5d}  → {dst}")


if __name__ == "__main__":
    main()
