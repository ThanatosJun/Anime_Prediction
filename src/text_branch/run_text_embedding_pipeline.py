"""
Run text embedding pipeline for Anime Prediction.

This script:
1) Loads split CSV files from data/processed
2) Cleans description text
3) Generates embeddings with Sentence-Transformers
4) Saves split-wise parquet artifacts
5) Writes a pipeline summary JSON report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

try:
    from .embedding_generator import EmbeddingGenerator
    from .text_preprocessor import TextPreprocessor
except ImportError:
    from embedding_generator import EmbeddingGenerator
    from text_preprocessor import TextPreprocessor


DEFAULT_CONFIG_PATH = Path("src/text_branch/configs/embedding_config.yaml")
DEFAULT_INPUT_TEMPLATE = "anilist_anime_multimodal_input_{split}.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text embedding pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to embedding config YAML.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing split CSV files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (e.g., train val test holdout_unknown).",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="description",
        help="Text column to clean and embed.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="ID column to preserve in output.",
    )
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=["popularity", "meanScore"],
        help="Target columns to preserve in output.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="text_embeddings",
        help="Output parquet prefix. Example: text_embeddings_train.parquet",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="text_embedding_pipeline_summary.json",
        help="Summary report filename under report_dir.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional row cap per split for quick debug runs. 0 means full split.",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_embedding_frame(
    df: pd.DataFrame,
    embedding_matrix: np.ndarray,
    embedding_dim: int,
) -> pd.DataFrame:
    emb_cols = [f"emb_{i:03d}" for i in range(embedding_dim)]
    emb_df = pd.DataFrame(embedding_matrix, index=df.index, columns=emb_cols)
    return emb_df


def _process_split(
    split: str,
    input_csv: Path,
    text_column: str,
    id_column: str,
    target_columns: List[str],
    preprocessor: TextPreprocessor,
    generator: EmbeddingGenerator,
    output_dir: Path,
    output_prefix: str,
    sample_size: int,
) -> Dict:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input file for split '{split}': {input_csv}")

    df = pd.read_csv(input_csv)
    if sample_size > 0:
        df = df.head(sample_size).copy()

    required_columns = [id_column, text_column]
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Split '{split}' missing required columns: {missing_required}"
        )

    safe_targets = [c for c in target_columns if c in df.columns]

    df["text_clean"] = df[text_column].apply(preprocessor.clean)
    df["has_text_clean"] = df["text_clean"].notna().astype(int)

    keep_mask = df["has_text_clean"] == 1
    clean_df = df.loc[keep_mask].copy()

    if clean_df.empty:
        out_path = output_dir / f"{output_prefix}_{split}.parquet"
        cols = [id_column, "split", text_column, "text_clean", "has_text_clean"] + safe_targets
        empty_out = df.loc[:, [c for c in cols if c in df.columns]].copy()
        empty_out["split"] = split
        empty_out.to_parquet(out_path, index=False)
        return {
            "split": split,
            "input_rows": int(len(df)),
            "encoded_rows": 0,
            "dropped_rows": int(len(df)),
            "retention_rate": 0.0,
            "output_path": str(out_path.as_posix()),
        }

    texts = clean_df["text_clean"].tolist()
    emb = generator.encode(texts, show_progress_bar=True)

    emb_df = _build_embedding_frame(clean_df, emb, generator.embedding_dim)

    output_cols = [id_column, text_column, "text_clean", "has_text_clean"] + safe_targets
    out_df = clean_df.loc[:, [c for c in output_cols if c in clean_df.columns]].copy()
    out_df["split"] = split

    out_df = pd.concat([out_df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    out_path = output_dir / f"{output_prefix}_{split}.parquet"
    out_df.to_parquet(out_path, index=False)

    input_rows = int(len(df))
    encoded_rows = int(len(out_df))
    dropped_rows = input_rows - encoded_rows

    return {
        "split": split,
        "input_rows": input_rows,
        "encoded_rows": encoded_rows,
        "dropped_rows": dropped_rows,
        "retention_rate": float(encoded_rows / input_rows) if input_rows > 0 else 0.0,
        "output_path": str(out_path.as_posix()),
    }


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)

    embedding_cfg = config.get("embedding", {})
    preprocess_cfg = config.get("preprocessing", {})
    output_cfg = config.get("output", {})

    artifact_dir = Path(output_cfg.get("artifact_dir", "artifacts"))
    report_dir = Path(output_cfg.get("report_dir", "reports"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = TextPreprocessor(
        lowercase=preprocess_cfg.get("lowercase", True),
        remove_urls=preprocess_cfg.get("remove_urls", True),
        remove_extra_whitespace=preprocess_cfg.get("remove_extra_whitespace", True),
        min_length=int(preprocess_cfg.get("min_length", 10)),
        max_length=int(preprocess_cfg.get("max_length", 512)),
    )

    generator = EmbeddingGenerator(
        model_name=embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=embedding_cfg.get("device", "auto"),
        batch_size=int(embedding_cfg.get("batch_size", 16)),
        random_seed=int(config.get("random_seed", 42)),
    )

    run_stats = []
    for split in args.splits:
        input_csv = args.data_dir / DEFAULT_INPUT_TEMPLATE.format(split=split)
        print(f"\n[Split: {split}] Reading {input_csv.as_posix()}")
        split_stats = _process_split(
            split=split,
            input_csv=input_csv,
            text_column=args.text_column,
            id_column=args.id_column,
            target_columns=args.target_columns,
            preprocessor=preprocessor,
            generator=generator,
            output_dir=artifact_dir,
            output_prefix=args.output_prefix,
            sample_size=args.sample_size,
        )
        run_stats.append(split_stats)
        print(
            "Saved {path} | encoded={enc}/{total} (retention={ret:.2%})".format(
                path=split_stats["output_path"],
                enc=split_stats["encoded_rows"],
                total=split_stats["input_rows"],
                ret=split_stats["retention_rate"],
            )
        )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.as_posix()),
        "data_dir": str(args.data_dir.as_posix()),
        "splits": args.splits,
        "text_column": args.text_column,
        "id_column": args.id_column,
        "target_columns": args.target_columns,
        "sample_size": args.sample_size,
        "model_info": generator.get_model_info(),
        "preprocessing": preprocess_cfg,
        "split_stats": run_stats,
    }

    report_path = report_dir / args.report_name
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary report written: {report_path.as_posix()}")


if __name__ == "__main__":
    main()
