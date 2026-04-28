"""
Generates text embeddings for the Fusion RAG pipeline.

Input:
  {meta_dir}/fusion_meta_clean_{split}.csv  — id + description column
  (meta_dir configured in src/fussion_branch/configs/text_process_config.yaml)

Output:
  {out_dir}/text_embeddings_{split}.parquet
    columns: id, emb_0 .. emb_383

Usage:
  conda activate animeprediction
  python -m src.fussion_branch.run_text_embedding
  python -m src.fussion_branch.run_text_embedding --splits train val test
"""
import argparse
from pathlib import Path

import yaml
import pandas as pd

from src.fussion_branch.text_embedding import TextEmbedder

_CFG_PATH = Path("src/fussion_branch/configs/text_process_config.yaml")


def _load_paths(cfg_path: Path = _CFG_PATH) -> tuple[Path, Path]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    return (
        Path(paths.get("meta_dir", "data/fussion")),
        Path(paths.get("out_dir",  "src/fussion_branch/embedding/text")),
    )


def run(splits: tuple = ("train", "val", "test")) -> None:
    embedder = TextEmbedder()
    META_DIR, OUT_DIR = _load_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  meta_dir : {META_DIR}")
    print(f"  out_dir  : {OUT_DIR}")

    for split in splits:
        print(f"\n{'='*50}")
        print(f"  Split: {split}")
        print(f"{'='*50}")

        meta_path = META_DIR / f"fusion_meta_clean_{split}.csv"
        if not meta_path.exists():
            print(f"  not found: {meta_path} — skipping")
            continue

        df = pd.read_csv(meta_path, usecols=["id", "description"])
        df["id"] = df["id"].astype(int)
        print(f"  total rows: {len(df)}  null description: {df['description'].isna().sum()}")

        df["text_clean"] = df["description"].apply(
            lambda x: embedder.preprocessor.clean(x) if pd.notna(x) else None
        )
        clean_df = df[df["text_clean"].notna()].reset_index(drop=True)
        dropped  = len(df) - len(clean_df)
        print(f"  after cleaning: {len(clean_df)} kept, {dropped} dropped (null/too short)")

        if clean_df.empty:
            print(f"  no valid descriptions — skipping")
            continue

        embeddings = embedder.generator.encode(
            clean_df["text_clean"].tolist(), show_progress_bar=True
        )

        emb_cols = [f"emb_{i}" for i in range(embedder.dim)]
        out_df = pd.concat(
            [clean_df[["id"]].reset_index(drop=True),
             pd.DataFrame(embeddings, columns=emb_cols)],
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
