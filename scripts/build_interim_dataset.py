"""
Build an interim dataset from raw AniList exports.

Cleaning scope:
- keep model-relevant columns
- enforce stable dtypes
- deduplicate by id
- apply baseline missing-value imputation
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

RAW_PICKLE = RAW_DIR / "anilist_anime_data_complete.pkl"
RAW_CSV = RAW_DIR / "anilist_anime_data_complete.csv"

OUTPUT_BASENAME = "anilist_anime_data_interim"

KEEP_COLUMNS = [
    "id",
    "title_romaji",
    "title_english",
    "title_native",
    "type",
    "format",
    "status",
    "season",
    "seasonYear",
    "episodes",
    "duration",
    "averageScore",
    "meanScore",
    "popularity",
    "favourites",
    "trending",
    "source",
    "countryOfOrigin",
    "isAdult",
    "startDate_year",
    "startDate_month",
    "startDate_day",
    "genres",
    "studios",
]

NUMERIC_COLUMNS = [
    "episodes",
    "duration",
    "averageScore",
    "meanScore",
    "popularity",
    "favourites",
    "trending",
    "seasonYear",
    "startDate_year",
    "startDate_month",
    "startDate_day",
]


def load_raw_dataset() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError("No supported raw dataset found in data/raw (expected PKL or CSV).")


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in KEEP_COLUMNS if col in df.columns]
    return df[available].copy()


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df


def deduplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "id" not in df.columns:
        return df, 0
    before = len(df)
    deduped = df.drop_duplicates(subset=["id"], keep="first")
    removed = before - len(deduped)
    return deduped, removed


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if "episodes" in df.columns:
        by_format = df.groupby("format")["episodes"].transform("median")
        df["episodes"] = df["episodes"].fillna(by_format)
        df["episodes"] = df["episodes"].fillna(df["episodes"].median())

    if "duration" in df.columns:
        by_format = df.groupby("format")["duration"].transform("median")
        df["duration"] = df["duration"].fillna(by_format)
        df["duration"] = df["duration"].fillna(df["duration"].median())

    if "averageScore" in df.columns:
        score_source = df["meanScore"] if "meanScore" in df.columns else None
        if score_source is not None:
            df["averageScore"] = df["averageScore"].fillna(score_source)
        df["averageScore"] = df["averageScore"].fillna(df["averageScore"].median())

    if "seasonYear" in df.columns and "startDate_year" in df.columns:
        df["seasonYear"] = df["seasonYear"].fillna(df["startDate_year"])

    return df


def add_release_date(df: pd.DataFrame) -> pd.DataFrame:
    if {"startDate_year", "startDate_month", "startDate_day"}.issubset(df.columns):
        date_df = df[["startDate_year", "startDate_month", "startDate_day"]].copy()
        date_df.columns = ["year", "month", "day"]
        df["release_date"] = pd.to_datetime(date_df, errors="coerce")
    return df


def write_outputs(df: pd.DataFrame, metadata: dict) -> tuple[Path, Path]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    date_suffix = datetime.now().strftime("%Y%m%d")

    csv_path = INTERIM_DIR / f"{OUTPUT_BASENAME}_{date_suffix}.csv"
    meta_path = INTERIM_DIR / f"{OUTPUT_BASENAME}_{date_suffix}_meta.json"

    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return csv_path, meta_path


def main() -> None:
    raw_df = load_raw_dataset()
    interim_df = select_columns(raw_df)
    interim_df = enforce_dtypes(interim_df)
    interim_df, removed_duplicates = deduplicate_rows(interim_df)
    interim_df = impute_missing_values(interim_df)
    interim_df = add_release_date(interim_df)

    metadata = {
        "rows": int(len(interim_df)),
        "columns": int(len(interim_df.columns)),
        "removed_duplicates": int(removed_duplicates),
        "missing_after_cleaning": {
            col: float(value)
            for col, value in interim_df.isna().mean().sort_values(ascending=False).to_dict().items()
        },
    }

    csv_path, meta_path = write_outputs(interim_df, metadata)
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
