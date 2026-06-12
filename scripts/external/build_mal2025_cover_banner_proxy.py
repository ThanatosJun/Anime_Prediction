"""Build MAL 2025 cover-as-banner proxy splits.

MAL 2025 provides local cover images but no true AniList-style banner image.
For a diagnostic external variant, this script copies cover embeddings into the
banner embedding slot while preserving the already prepared YOLO embeddings.
The output must be described as a proxy, not as a real banner-image evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ["mal2025_popularity_local_ready_yolo", "mal2025_dual_local_ready_yolo"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MAL 2025 cover-as-banner proxy image splits.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--suffix", default="coverbanner")
    return parser.parse_args()


def _copy_sidecars(split: str, new_split: str) -> None:
    copies = [
        (
            ROOT / "src_2" / "data" / "dataset" / f"fusion_meta_clean_{split}_v2.csv",
            ROOT / "src_2" / "data" / "dataset" / f"fusion_meta_clean_{new_split}_v2.csv",
        ),
        (
            ROOT / "src_2" / "RAG" / "return" / f"rag_features_{split}.parquet",
            ROOT / "src_2" / "RAG" / "return" / f"rag_features_{new_split}.parquet",
        ),
        (
            ROOT / "src_2" / "embedding" / "text" / f"text_embeddings_{split}.parquet",
            ROOT / "src_2" / "embedding" / "text" / f"text_embeddings_{new_split}.parquet",
        ),
        (
            ROOT / "data" / "external_transformed" / f"{split}_id_map.csv",
            ROOT / "data" / "external_transformed" / f"{new_split}_id_map.csv",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())


def _build_split(split: str, suffix: str) -> dict:
    image_dir = ROOT / "src_2" / "embedding" / "image"
    src_path = image_dir / f"image_embeddings_{split}.parquet"
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    new_split = f"{split}_{suffix}"
    dst_path = image_dir / f"image_embeddings_{new_split}.parquet"
    df = pd.read_parquet(src_path)
    cover_cols = [f"cover_{idx}" for idx in range(1024)]
    banner_cols = [f"banner_{idx}" for idx in range(1024)]
    missing = [col for col in cover_cols + banner_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{src_path} is missing expected embedding columns: {missing[:5]}")

    out = df.copy()
    out.loc[:, banner_cols] = out.loc[:, cover_cols].to_numpy()
    if "has_banner" in out.columns:
        out["has_banner"] = 1
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst_path, index=False)
    _copy_sidecars(split, new_split)

    return {
        "source_split": split,
        "new_split": new_split,
        "rows": int(len(out)),
        "cover_nonzero_rows": int((out[cover_cols].abs().sum(axis=1) > 0).sum()),
        "banner_nonzero_rows": int((out[banner_cols].abs().sum(axis=1) > 0).sum()),
        "output": dst_path.relative_to(ROOT).as_posix(),
        "claim_boundary": "Cover embeddings copied into banner slot; diagnostic proxy only.",
    }


def main() -> None:
    args = _parse_args()
    summaries = [_build_split(split, args.suffix) for split in args.splits]
    summary = {"splits": summaries}
    summary_path = ROOT / "data" / "external_transformed" / f"mal2025_cover_banner_proxy_summary_{args.suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
