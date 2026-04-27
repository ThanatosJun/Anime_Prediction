"""
Generate image embeddings for all splits using fine-tuned Swin-base.

Input:
  data/fussion/fusion_meta_clean_{split}.csv  — valid IDs per split
  data/image/{split_dir}/{id}_coverImage_medium.jpg

Output:
  src/fussion_branch/embedding/image/image_embeddings_{split}.parquet
    columns: id, img_0 .. img_1023

Usage:
  conda activate animeprediction
  python -m src.fussion_branch.run_image_embedding
  python -m src.fussion_branch.run_image_embedding --checkpoint results/01/best
  python -m src.fussion_branch.run_image_embedding --splits train val
"""
import argparse
from pathlib import Path

import pandas as pd

from src.fussion_branch.image_embedding import ImageEmbedder

FUSION_META_DIR = Path("data/fussion")
IMAGE_BASE_DIR  = Path("data/image")
OUT_DIR         = Path("src/fussion_branch/embedding/image")

SPLIT_IMAGE_DIR = {
    "train":           "train_image",
    "val":             "validation_image",
    "test":            "test_image",
    "holdout_unknown": "holdout_unknow_image",
}


def run(
    splits: tuple = ("train", "val", "test", "holdout_unknown"),
    checkpoint_dir: str = "results/01/best",
    batch_size: int = 64,
) -> None:
    embedder = ImageEmbedder(checkpoint_dir=checkpoint_dir)
    print(f"Model dim: {embedder.dim}  device: {embedder.device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        meta_path = FUSION_META_DIR / f"fusion_meta_clean_{split}.csv"
        if not meta_path.exists():
            print(f"[{split}] fusion_meta_clean not found — skipping")
            continue

        ids = pd.read_csv(meta_path, usecols=["id"])["id"].astype(int).tolist()
        img_dir = IMAGE_BASE_DIR / SPLIT_IMAGE_DIR[split]

        paths = [
            str(img_dir / f"{anime_id}_coverImage_medium.jpg")
            for anime_id in ids
        ]

        missing = sum(1 for p in paths if not Path(p).exists())
        print(f"[{split}] {len(ids)} anime  |  images missing: {missing}")

        embs = embedder.encode_paths(paths, batch_size=batch_size)  # (N, 1024)

        img_cols = [f"img_{j}" for j in range(embs.shape[1])]
        df = pd.DataFrame(embs, columns=img_cols)
        df.insert(0, "id", ids)

        out_path = OUT_DIR / f"image_embeddings_{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[{split}] → {out_path}  shape={df.shape}")

    print("\nImage embedding generation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image embeddings for Fusion MLP")
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "val", "test", "holdout_unknown"],
    )
    parser.add_argument(
        "--checkpoint", default="results/01/best",
        help="Fine-tuned Swin checkpoint dir (default: results/01/best)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    run(splits=tuple(args.splits), checkpoint_dir=args.checkpoint, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
