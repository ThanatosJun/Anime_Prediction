"""
Export multimodal-ready model inputs with physical split files.

Inputs:
- data/raw/anilist_anime_data_complete.pkl or .csv
- data/processed/anilist_anime_data_processed_v1.csv

Outputs:
- data/processed/anilist_anime_multimodal_input_v1.csv
- data/processed/anilist_anime_multimodal_input_{train|val|test|holdout_unknown}.csv
- data/eda/multimodal_input_summary.json
- data/eda/multimodal_input_summary.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EDA_DIR = Path("data/eda")

RAW_PICKLE = RAW_DIR / "anilist_anime_data_complete.pkl"
RAW_CSV = RAW_DIR / "anilist_anime_data_complete.csv"
PROCESSED_CSV = PROCESSED_DIR / "anilist_anime_data_processed_v1.csv"

MULTIMODAL_CSV = PROCESSED_DIR / "anilist_anime_multimodal_input_v1.csv"
MULTIMODAL_SPLIT_TEMPLATE = "anilist_anime_multimodal_input_{split}.csv"
MULTIMODAL_SUMMARY_JSON = EDA_DIR / "multimodal_input_summary.json"
MULTIMODAL_SUMMARY_MD = EDA_DIR / "multimodal_input_summary.md"

RAW_MODALITY_COLUMNS = [
    "id",
    "title_romaji",
    "title_english",
    "description",
    "coverImage_medium",
    "bannerImage",
    "trailer_id",
    "trailer_site",
    "trailer_thumbnail",
]
PROCESSED_TARGET_COLUMNS = [
    "id",
    "release_year",
    "release_quarter",
    "split_pre_release_effective",
    "is_model_split",
    "popularity",
    "meanScore",
    "popularity_quarter_pct",
    "popularity_quarter_bucket",
]


def _load_raw() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError("No supported raw dataset found in data/raw (expected PKL or CSV).")


def _load_processed() -> pd.DataFrame:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError("Processed dataset not found. Run scripts/build_processed_dataset.py first.")
    return pd.read_csv(PROCESSED_CSV)


def _safe_col_select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    selected = [c for c in columns if c in df.columns]
    return df[selected].copy()


def _with_availability_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["has_text_description"] = df["description"].notna() if "description" in df.columns else False
    df["has_cover_image"] = df["coverImage_medium"].notna() if "coverImage_medium" in df.columns else False
    df["has_banner_image"] = df["bannerImage"].notna() if "bannerImage" in df.columns else False
    df["has_trailer"] = df["trailer_id"].notna() if "trailer_id" in df.columns else False
    return df


def _availability_ratio(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    return float(df[column].mean())


def _build_summary(df: pd.DataFrame) -> dict:
    split_counts = (
        df["split_pre_release_effective"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
        if "split_pre_release_effective" in df.columns
        else pd.DataFrame(columns=["split", "count"])
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "feature_contract": {
            "join_key": "id",
            "target_columns": [c for c in PROCESSED_TARGET_COLUMNS if c in df.columns],
            "raw_multimodal_columns": [c for c in RAW_MODALITY_COLUMNS if c in df.columns],
            "availability_flags": [
                "has_text_description",
                "has_cover_image",
                "has_banner_image",
                "has_trailer",
            ],
        },
        "availability_ratios": {
            "has_text_description": _availability_ratio(df, "has_text_description"),
            "has_cover_image": _availability_ratio(df, "has_cover_image"),
            "has_banner_image": _availability_ratio(df, "has_banner_image"),
            "has_trailer": _availability_ratio(df, "has_trailer"),
        },
        "split_counts": [
            {"split": str(row["split"]), "count": int(row["count"])}
            for _, row in split_counts.iterrows()
        ],
    }
    return summary


def _write_summary(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    MULTIMODAL_SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Multimodal Input Export Summary",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Columns: `{summary['column_count']}`",
        "",
        "## Feature Contract",
        "",
        f"- Join key: `{summary['feature_contract']['join_key']}`",
        f"- Target columns: `{summary['feature_contract']['target_columns']}`",
        f"- Raw multimodal columns: `{summary['feature_contract']['raw_multimodal_columns']}`",
        f"- Availability flags: `{summary['feature_contract']['availability_flags']}`",
        "",
        "## Modality Coverage",
        "",
    ]
    for key, value in summary["availability_ratios"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Split Counts", ""])
    for item in summary["split_counts"]:
        lines.append(f"- `{item['split']}`: {item['count']}")
    MULTIMODAL_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_df = _load_raw()
    processed_df = _load_processed()

    raw_subset = _safe_col_select(raw_df, RAW_MODALITY_COLUMNS)
    processed_subset = _safe_col_select(processed_df, PROCESSED_TARGET_COLUMNS)
    merged = processed_subset.merge(raw_subset, on="id", how="left")
    merged = _with_availability_flags(merged)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MULTIMODAL_CSV, index=False)

    if "split_pre_release_effective" in merged.columns:
        for split_name, group in merged.groupby("split_pre_release_effective", dropna=False):
            name = "unknown" if pd.isna(split_name) else str(split_name)
            out_path = PROCESSED_DIR / MULTIMODAL_SPLIT_TEMPLATE.format(split=name)
            group.to_csv(out_path, index=False)

    summary = _build_summary(merged)
    _write_summary(summary)

    print(f"Wrote {MULTIMODAL_CSV}")
    print(f"Wrote {MULTIMODAL_SUMMARY_JSON}")
    print(f"Wrote {MULTIMODAL_SUMMARY_MD}")


if __name__ == "__main__":
    main()
