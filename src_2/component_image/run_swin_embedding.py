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

from model import load_model, get_embedding, get_stage_embeddings
from image_process import load_image, ResizeWithPad, get_transform_original
from config import load_config

_CFG_PATH = str(_HERE / "image_encoder_config.yaml")

_META_DIR  = _ROOT / "src_2" / "data" / "dataset"
_IMAGE_DIR = _ROOT / "src_2" / "data" / "image"
_CROP_DIR  = _ROOT / "src_2" / "data" / "image" / "crops"
_EMB_ROOT  = _ROOT / "src_2" / "embedding"

# embed mode → 每個模態的維度
#   pooler : Swin pooler_output                          → 1024
#   stage  : 4 個 stage（128+256+512+1024）concat        → 1920
_MODE_DIM = {"pooler": 1024, "stage": 1920}
# embed mode → 輸出子目錄（pooler / stage 並存，互不覆蓋）
_MODE_DIR = {"pooler": "image", "stage": "image_stage"}

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


def _embed(model, imgs, resize, transform, device, mode: str = "pooler") -> np.ndarray:
    """多張圖 mean pool → 單一向量。mode=stage 時 concat 前 4 個 stage（1920-dim）。

    註：Swin-B 的 reshaped_hidden_states 實際有 5 個 [128,256,512,1024,1024]，
    第 5 個是最後 stage 的 final-norm 版（與第 4 個 cosine≈0.89 高度重複），故只取前 4 個。
    """
    batch = torch.stack([transform(resize(i)) for i in imgs]).to(device)
    with torch.no_grad():
        if mode == "stage":
            stages = get_stage_embeddings(model, batch)[:4]        # [(N,128),(N,256),(N,512),(N,1024)]
            return torch.cat([s.mean(dim=0) for s in stages], dim=0).cpu().numpy()  # (1920,)
        return get_embedding(model, batch).mean(dim=0).cpu().numpy()                # (1024,)


def run_swin_embedding(splits: list, mode: str = None):
    config = load_config(_CFG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_swin(config, device)

    # mode：CLI 優先，否則讀 config（fusion_embed_mode）
    if mode is None:
        mode = config.get("model", {}).get("fusion_embed_mode", "pooler")
    emb_dim = _MODE_DIM[mode]
    zero    = lambda: [0.0] * emb_dim

    # pooler / stage 分目錄並存，互不覆蓋
    out_dir = _EMB_ROOT / _MODE_DIR[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Embed mode: {mode}  (dim={emb_dim} per modality)  →  {out_dir.relative_to(_ROOT)}")

    image_size = config["data"]["image_size"]
    resize     = ResizeWithPad(image_size)
    transform  = get_transform_original(image_size)

    yolo_cols   = [f"yolo_{i}"   for i in range(emb_dim)]
    cover_cols  = [f"cover_{i}"  for i in range(emb_dim)]
    banner_cols = [f"banner_{i}" for i in range(emb_dim)]

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
                r.update(dict(zip(yolo_cols, _embed(model, imgs, resize, transform, device, mode).tolist())))
                r["has_yolo"] = 1
            else:
                r.update(dict(zip(yolo_cols, zero())))
                r["has_yolo"] = 0
                stats["no_yolo"] += 1

            # ── cover ─────────────────────────────────────────────────────────
            cover_file = img_dir / f"{anime_id}_coverImage_extraLarge.jpg"
            img = load_image(str(cover_file)) if cover_file.exists() else None
            if img:
                r.update(dict(zip(cover_cols, _embed(model, [img], resize, transform, device, mode).tolist())))
                r["has_cover"] = 1
            else:
                r.update(dict(zip(cover_cols, zero())))
                r["has_cover"] = 0
                stats["no_cover"] += 1

            # ── banner ────────────────────────────────────────────────────────
            banner_file = img_dir / f"{anime_id}_bannerImage.jpg"
            img = load_image(str(banner_file)) if banner_file.exists() else None
            if img:
                r.update(dict(zip(banner_cols, _embed(model, [img], resize, transform, device, mode).tolist())))
                r["has_banner"] = 1
            else:
                r.update(dict(zip(banner_cols, zero())))
                r["has_banner"] = 0
                stats["no_banner"] += 1

            rows.append(r)

        out_df   = pd.DataFrame(rows)
        out_path = out_dir / f"image_embeddings_{split}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(out_df)} rows → {out_path.relative_to(_ROOT)}")
        print(f"  no_yolo={stats['no_yolo']}  no_cover={stats['no_cover']}  no_banner={stats['no_banner']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--mode", choices=["pooler", "stage"], default=None,
                        help="覆蓋 config 的 fusion_embed_mode；pooler→embedding/image/，stage→embedding/image_stage/")
    args = parser.parse_args()
    run_swin_embedding(args.splits, mode=args.mode)


if __name__ == "__main__":
    main()
