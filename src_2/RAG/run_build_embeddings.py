"""
為 RAG knowledge base 生成 text + image embeddings

text：所有 splits（train/val/test/holdout_unknown）
image：僅 TRAIN（RAG knowledge base 只需訓練集）

輸出：
  src_2/embedding/text/text_embeddings_{split}.parquet   → id, emb_000 … emb_767
  src_2/embedding/image_rag/image_embeddings_train.parquet → id, img_0  … img_1023

用法：
  python run_build_embeddings.py                          # text 全 splits + image train
  python run_build_embeddings.py --modality text          # 只跑 text
  python run_build_embeddings.py --modality image         # 只跑 image（train）
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
sys.path.insert(0, str(_ROOT / "src_2" / "component_text"))

from output import TextEmbedder     # noqa: E402  (component_text)


_META_DIR   = _ROOT / "src_2" / "data" / "dataset"
_IMAGE_DIR  = _ROOT / "src_2" / "data" / "image"
_TEXT_OUT   = _ROOT / "src_2" / "embedding" / "text"
_IMAGE_OUT  = _ROOT / "src_2" / "embedding" / "image_rag"
_TEXT_OUT.mkdir(parents=True, exist_ok=True)
_IMAGE_OUT.mkdir(parents=True, exist_ok=True)

_IMAGE_SPLIT_DIR = {
    "train":           "train_image",
    "val":             "validation_image",
    "test":            "test_image",
    "holdout_unknown": "holdout_unknow_image",
}


# ── text embeddings ───────────────────────────────────────────────────────────

def build_text_embeddings(splits: list):
    embedder = TextEmbedder()  # config.py 預設從 __file__ 旁的 yaml 載入
    dim = embedder.embedding_dim  # 768

    for split in splits:
        csv_path = _META_DIR / f"fusion_meta_clean_{split}_v2.csv"
        if not csv_path.exists():
            print(f"[text] {csv_path.name} not found — skip")
            continue

        out_path = _TEXT_OUT / f"text_embeddings_{split}.parquet"
        df = pd.read_csv(csv_path)
        print(f"\n[text/{split}] {len(df)} rows")

        id_to_emb = embedder.embed_dataframe(df, text_col="description", id_col="id")

        emb_cols = [f"emb_{i:03d}" for i in range(dim)]
        rows = []
        for anime_id, emb in id_to_emb.items():
            row = {"id": anime_id}
            row.update(dict(zip(emb_cols, emb.tolist())))
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(out_df)} embeddings → {out_path.relative_to(_ROOT)}")


# ── image embeddings ──────────────────────────────────────────────────────────

def build_image_embeddings(splits: list):
    import importlib.util
    import yaml

    # component_image 插到 sys.path 最前面，確保 output.py 內的 from config import 找到正確模組
    _img_dir = str(Path(__file__).parent.parent / "component_image")
    if _img_dir not in sys.path:
        sys.path.insert(0, _img_dir)

    _spec = importlib.util.spec_from_file_location(
        "image_output",
        Path(_img_dir) / "output.py",
    )
    _img_mod = importlib.util.module_from_spec(_spec)

    # 清除 sys.modules 中的 'config' cache，避免 output.py 找到 component_text 的 config
    sys.modules.pop('config', None)
    _spec.loader.exec_module(_img_mod)

    _ImageEmb = _img_mod.ImageEmbedder

    cfg_path = Path(__file__).parent.parent / "component_image" / "image_encoder_config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    model_path = str(_ROOT / cfg["inference"]["model_path"])
    embedder = _ImageEmb(model_path=model_path, config=cfg)
    embedder.use_yolo = False   # RAG 使用 raw cover image，不做 YOLO crop
    dim = 1024

    emb_cols = [f"img_{i}" for i in range(dim)]

    for split in splits:
        csv_path = _META_DIR / f"fusion_meta_clean_{split}_v2.csv"
        if not csv_path.exists():
            print(f"[image] {csv_path.name} not found — skip")
            continue

        img_dir = _IMAGE_DIR / _IMAGE_SPLIT_DIR.get(split, f"{split}_image")
        if not img_dir.exists():
            print(f"[image/{split}] image dir not found: {img_dir} — skip")
            continue

        out_path = _IMAGE_OUT / f"image_embeddings_{split}.parquet"
        df = pd.read_csv(csv_path)
        print(f"\n[image/{split}] {len(df)} rows, image dir: {img_dir.name}")

        rows = []
        skipped = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"image/{split}"):
            anime_id = int(row["id"])
            img_file = img_dir / f"{anime_id}_coverImage_extraLarge.jpg"
            if not img_file.exists():
                skipped += 1
                continue

            emb = embedder.embed(str(img_file))
            if emb is None:
                skipped += 1
                continue

            r = {"id": anime_id}
            r.update(dict(zip(emb_cols, emb.tolist())))
            rows.append(r)

        out_df = pd.DataFrame(rows)
        out_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(out_df)} embeddings (skipped={skipped}) → {out_path.relative_to(_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-splits", nargs="+",
                        default=["train", "val", "test", "holdout_unknown"],
                        help="text embedding splits（預設全部）")
    parser.add_argument("--modality", choices=["text", "image", "both"], default="both")
    args = parser.parse_args()

    if args.modality in ("text", "both"):
        print("=== Text Embeddings（all splits）===")
        build_text_embeddings(args.text_splits)

    if args.modality in ("image", "both"):
        print("\n=== Image Embeddings（RAG: train only）===")
        build_image_embeddings(["train"])   # RAG knowledge base 只需 train


if __name__ == "__main__":
    main()
