"""
Step 2: Encode YOLO crops with Swin (batched) and mean-pool per anime.

Input:
  data/image/crops/crop_manifest_{split}.parquet
    columns: id (int), crop_path (str)   — produced by run_yolo_crop.py

Output:
  src/fussion_branch/embedding/image/image_embeddings_char_{split}.parquet
    columns: id, img_0 .. img_1023

Usage:
  conda activate animeprediction
  python -m src.fussion_branch.run_char_embedding
  python -m src.fussion_branch.run_char_embedding --splits train val
  python -m src.fussion_branch.run_char_embedding --checkpoint src/fussion_branch/model/best
"""
import argparse
from pathlib import Path

import pandas as pd

from src.fussion_branch.image_embedding import CharacterEmbedder

CROPS_BASE_DIR = Path("data/image/crops")
OUT_DIR        = Path("src/fussion_branch/embedding/image")


def run(
    splits: tuple = ("train", "val", "test", "holdout_unknown"),
    checkpoint_dir: str = "src/fussion_branch/model/best",
    batch_size: int = 64,
) -> None:
    embedder = CharacterEmbedder(checkpoint_dir=checkpoint_dir)
    print(f"dim: {embedder.dim}  device: {embedder.device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        manifest_path = CROPS_BASE_DIR / f"crop_manifest_{split}.parquet"
        if not manifest_path.exists():
            print(f"[{split}] crop_manifest not found — run run_yolo_crop.py first")
            continue

        manifest = pd.read_parquet(manifest_path)
        print(f"[{split}] {manifest['id'].nunique()} anime  |  {len(manifest)} crops")

        ids, embs = embedder.encode_manifest(manifest, batch_size=batch_size)

        df = pd.DataFrame(embs, columns=[f"img_{j}" for j in range(embs.shape[1])])
        df.insert(0, "id", ids)

        out_path = OUT_DIR / f"image_embeddings_char_{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[{split}] → {out_path}  shape={df.shape}")

    print("\nCharacter embedding generation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode YOLO crops with Swin (batched)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "holdout_unknown"])
    parser.add_argument("--checkpoint", default="src/fussion_branch/model/best")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    run(splits=tuple(args.splits), checkpoint_dir=args.checkpoint, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
