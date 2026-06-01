"""
Step 1：YOLO 偵測並儲存 crop 圖片

輸入：src_2/data/image/{split}_image/{id}_coverImage_extraLarge.jpg
輸出：src_2/data/image/crops/{split}/{id}_crop_{n}.jpg
      （n = 0-based，依信心分數由高到低排序）
      無偵測結果時 fallback 儲存整張圖為 {id}_crop_0.jpg

用法：
    python run_yolo_crop.py
    python run_yolo_crop.py --splits train val test holdout_unknown
    python run_yolo_crop.py --splits train --detect-mode face
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))

from YOLO import detect_person, detect_faces
from image_process import load_image
from config import load_config, load_yolo_config

_CFG_PATH = str(_HERE / "image_encoder_config.yaml")

_META_DIR  = _ROOT / "src_2" / "data" / "dataset"
_IMAGE_DIR = _ROOT / "src_2" / "data" / "image"
_CROP_DIR  = _ROOT / "src_2" / "data" / "image" / "crops"

_SPLIT_IMAGE_DIR = {
    "train":           "train_image",
    "val":             "validation_image",
    "test":            "test_image",
    "holdout_unknown": "holdout_unknow_image",
}


def _get_crops(img: Image.Image, yolo_cfg: dict, detect_mode: str):
    """YOLO 偵測並回傳 (crop_list, fallback:bool)"""
    w, h = img.size
    scale = max(640 / w, 640 / h)
    if scale > 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    results = []
    if detect_mode in ('person', 'both'):
        m, d = yolo_cfg['model'], yolo_cfg['detection']
        results += detect_person(
            img, level=m['level'], version=m['version'],
            conf_threshold=d['conf_threshold'], iou_threshold=d['iou_threshold'],
        )
    if detect_mode in ('face', 'both'):
        m, d = yolo_cfg['face_model'], yolo_cfg['face_detection']
        results += detect_faces(
            img, level=m['level'], version=m['version'],
            conf_threshold=d['conf_threshold'], iou_threshold=d['iou_threshold'],
        )

    det_key  = 'face_detection' if detect_mode == 'face' else 'detection'
    max_det  = yolo_cfg[det_key]['max_detections']
    results  = sorted(results, key=lambda x: x[2], reverse=True)[:max_det]

    if not results:
        return [img], True   # fallback 整張圖
    return [img.crop(bbox) for (bbox, _, _) in results], False


def run_yolo_crop(splits: list, detect_mode: str):
    yolo_cfg = load_yolo_config(_CFG_PATH)
    # CLI 傳入的 detect_mode 優先
    effective_mode = detect_mode or yolo_cfg.get('detect_mode', 'face')

    for split in splits:
        csv_path = _META_DIR / f"fusion_meta_clean_{split}_v2.csv"
        if not csv_path.exists():
            print(f"[{split}] CSV not found — skip")
            continue

        img_dir  = _IMAGE_DIR / _SPLIT_IMAGE_DIR.get(split, f"{split}_image")
        crop_dir = _CROP_DIR / split
        crop_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)
        saved = fallback = skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"YOLO crop [{split}]"):
            anime_id  = int(row["id"])
            img_file  = img_dir / f"{anime_id}_coverImage_extraLarge.jpg"
            if not img_file.exists():
                skipped += 1
                continue

            img = load_image(str(img_file))
            if img is None:
                skipped += 1
                continue

            crops, is_fallback = _get_crops(img, yolo_cfg, effective_mode)
            if is_fallback:
                fallback += 1

            for n, crop in enumerate(crops):
                out_file = crop_dir / f"{anime_id}_crop_{n}.jpg"
                crop.save(str(out_file), "JPEG", quality=95)
                saved += 1

        print(f"  [{split}] saved={saved}  fallback={fallback}  skipped={skipped}")
        print(f"  crop dir → {crop_dir.relative_to(_ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test", "holdout_unknown"])
    parser.add_argument("--detect-mode", choices=["face", "person", "both"],
                        default=None, help="覆蓋 config 的 detect_mode")
    args = parser.parse_args()
    run_yolo_crop(args.splits, args.detect_mode)


if __name__ == "__main__":
    main()
