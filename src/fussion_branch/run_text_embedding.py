"""
Generates text embeddings for the Fusion RAG pipeline.

Input:
  data/fussion/fusion_meta_clean_{split}.csv              — valid IDs per split
  data/processed/anilist_anime_multimodal_input_{split}.csv — description column

Output:
  src/fussion_branch/RAG/text_embeddings_{split}.parquet
    columns: id, emb_000 .. emb_383

Usage:
  conda activate animeprediction
  python -m src.fussion_branch.run_text_embedding
  python -m src.fussion_branch.run_text_embedding --splits train val test
"""
import argparse
from pathlib import Path

import pandas as pd

from src.fussion_branch.text_embedding import TextEmbedder

FUSION_META_DIR = Path("data/fussion")
MULTIMODAL_DIR  = Path("data/processed")
OUT_DIR         = Path("src/fussion_branch/RAG")


def run(splits: tuple = ("train", "val", "test")) -> None:
    embedder = TextEmbedder()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n{'='*50}")
        print(f"  Split: {split}")
        print(f"{'='*50}")

        # Step 1: valid IDs from fusion_meta_clean
        meta_path = FUSION_META_DIR / f"fusion_meta_clean_{split}.csv"
        if not meta_path.exists():
            print(f"  fusion_meta_clean not found: {meta_path} — skipping")
            continue
        valid_ids = set(pd.read_csv(meta_path, usecols=["id"])["id"].astype(int))
        print(f"  fusion_meta_clean IDs: {len(valid_ids)}")

        # Step 2: load descriptions, filter to valid IDs
        input_path = MULTIMODAL_DIR / f"anilist_anime_multimodal_input_{split}.csv"
        if not input_path.exists():
            print(f"  description file not found: {input_path} — skipping")
            continue
        desc_df = pd.read_csv(input_path, usecols=["id", "description"])
        desc_df["id"] = desc_df["id"].astype(int)
        desc_df = desc_df[desc_df["id"].isin(valid_ids)].copy().reset_index(drop=True)
        print(f"  descriptions matched: {len(desc_df)}/{len(valid_ids)}")

        # Step 3: clean text, keep only valid rows
        desc_df["text_clean"] = desc_df["description"].apply(embedder.preprocessor.clean)
        clean_df = desc_df[desc_df["text_clean"].notna()].copy().reset_index(drop=True)
        dropped  = len(desc_df) - len(clean_df)
        print(f"  after cleaning: {len(clean_df)} kept, {dropped} dropped (null/too short)")

        if clean_df.empty:
            print(f"  no valid descriptions — skipping")
            continue

        # Step 4: generate embeddings
        embeddings = embedder.generator.encode(clean_df["text_clean"].tolist(), show_progress_bar=True)

        # Step 5: build flat parquet: id + emb_000..emb_383
        emb_cols = [f"emb_{i:03d}" for i in range(embedder.dim)]
        out_df = pd.concat(
            [
                clean_df[["id"]].reset_index(drop=True),
                pd.DataFrame(embeddings, columns=emb_cols),
            ],
            axis=1,
        )

        out_path = OUT_DIR / f"text_embeddings_{split}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  → {out_path}  shape={out_df.shape}")

    print("\nText embedding generation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text embeddings for Fusion RAG")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    args = parser.parse_args()
    run(splits=tuple(args.splits))


if __name__ == "__main__":
    main()
