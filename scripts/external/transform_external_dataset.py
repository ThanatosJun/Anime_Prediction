"""
Transform external anime datasets into project-compatible processed/multimodal contracts.

Usage example:
python scripts/external/transform_external_dataset.py ^
  --input-csv data/external/new_snapshot.csv ^
  --mapping-json docs/pipeline/external_schema_mapping_example.json ^
  --output-prefix external_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/external_transformed")

SEASON_TO_QUARTER = {"WINTER": 1, "SPRING": 2, "SUMMER": 3, "FALL": 4}
CLIP_COLUMNS = {
    "episodes": (0.01, 0.99),
    "duration": (0.01, 0.99),
    "averageScore": (0.005, 0.995),
    "meanScore": (0.005, 0.995),
    "popularity": (0.01, 0.99),
    "favourites": (0.01, 0.99),
    "trending": (0.01, 0.95),
}
LOWER_BOUND_COLUMNS = ["episodes", "duration", "averageScore", "meanScore", "popularity", "favourites", "trending"]

PROCESSED_REQUIRED_COLUMNS = [
    "id",
    "title_romaji",
    "title_english",
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
    "voice_actor_names",
]

MULTIMODAL_CONTRACT_COLUMNS = [
    "id",
    "release_year",
    "release_quarter",
    "split_pre_release_effective",
    "is_model_split",
    "popularity",
    "meanScore",
    "popularity_quarter_pct",
    "popularity_quarter_bucket",
    "title_romaji",
    "title_english",
    "description",
    "coverImage_medium",
    "bannerImage",
    "trailer_id",
    "trailer_site",
    "trailer_thumbnail",
    "has_text_description",
    "has_cover_image",
    "has_banner_image",
    "has_trailer",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform external dataset to project contracts.")
    parser.add_argument("--input-csv", required=True, help="Path to external source CSV.")
    parser.add_argument(
        "--mapping-json",
        required=True,
        help="Path to JSON map: {\"external_col\": \"project_col\"}.",
    )
    parser.add_argument(
        "--output-prefix",
        default="external_v1",
        help="Output file prefix under data/external_transformed.",
    )
    return parser.parse_args()


def _read_mapping(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mapping-json must be an object of external->project column names.")
    return {str(k): str(v) for k, v in data.items()}


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _apply_interim_like_imputation(df: pd.DataFrame) -> None:
    if "episodes" in df.columns:
        by_format = df.groupby("format")["episodes"].transform("median")
        df["episodes"] = df["episodes"].fillna(by_format).fillna(df["episodes"].median())

    if "duration" in df.columns:
        by_format = df.groupby("format")["duration"].transform("median")
        df["duration"] = df["duration"].fillna(by_format).fillna(df["duration"].median())

    if "averageScore" in df.columns:
        if "meanScore" in df.columns:
            df["averageScore"] = df["averageScore"].fillna(df["meanScore"])
        df["averageScore"] = df["averageScore"].fillna(df["averageScore"].median())

    if "seasonYear" in df.columns and "startDate_year" in df.columns:
        df["seasonYear"] = df["seasonYear"].fillna(df["startDate_year"])

    if "title_english" in df.columns and "title_romaji" in df.columns:
        df["title_english"] = df["title_english"].fillna(df["title_romaji"])

    if "voice_actor_names" in df.columns:
        df["voice_actor_names"] = df["voice_actor_names"].fillna("")


def _apply_processed_like_rules(df: pd.DataFrame) -> None:
    for col in LOWER_BOUND_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    for col, (q_low, q_high) in CLIP_COLUMNS.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue
        lower = float(clean.quantile(q_low))
        upper = float(clean.quantile(q_high))
        df[col] = series.clip(lower=lower, upper=upper)

    season_series = (
        df["season"].astype(str).str.upper().str.strip()
        if "season" in df.columns
        else pd.Series(index=df.index, dtype="object")
    )
    season_quarter = season_series.map(SEASON_TO_QUARTER)
    month_quarter = pd.Series(index=df.index, dtype="float64")
    if "startDate_month" in df.columns:
        month_vals = pd.to_numeric(df["startDate_month"], errors="coerce")
        month_quarter = ((month_vals - 1) // 3 + 1).where(month_vals.between(1, 12))
    df["release_quarter"] = season_quarter.fillna(month_quarter).astype("Int64")

    release_year = pd.Series(index=df.index, dtype="Int64")
    if "seasonYear" in df.columns:
        release_year = pd.to_numeric(df["seasonYear"], errors="coerce").astype("Int64")
    if "startDate_year" in df.columns:
        release_year = release_year.fillna(pd.to_numeric(df["startDate_year"], errors="coerce").astype("Int64"))
    df["release_year"] = release_year

    valid = df["release_year"].notna() & df["release_quarter"].notna()
    df["release_quarter_key"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[valid, "release_quarter_key"] = (
        df.loc[valid, "release_year"].astype(int).astype(str)
        + "Q"
        + df.loc[valid, "release_quarter"].astype(int).astype(str)
    )

    if "popularity" in df.columns:
        pct = pd.Series(pd.NA, index=df.index, dtype="Float64")
        pct.loc[valid] = (
            pd.to_numeric(df.loc[valid, "popularity"], errors="coerce")
            .groupby(df.loc[valid, "release_quarter_key"], dropna=False)
            .rank(pct=True, ascending=True)
            .astype("Float64")
        )
        df["popularity_quarter_pct"] = pct
        bins = [-0.000001, 0.25, 0.50, 0.75, 1.0]
        labels = ["cold_0_25", "warm_25_50", "hot_50_75", "top_75_100"]
        df["popularity_quarter_bucket"] = pd.cut(pct, bins=bins, labels=labels)
        df.loc[pct.isna(), "popularity_quarter_bucket"] = pd.NA
    else:
        df["popularity_quarter_pct"] = pd.NA
        df["popularity_quarter_bucket"] = pd.NA

    df["split_pre_release_effective"] = "inference_only"
    df["is_model_split"] = False


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    out = df[available].copy()
    for col in missing:
        out[col] = pd.NA
    return out[columns]


def _with_multimodal_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_text_description"] = df["description"].notna() if "description" in df.columns else False
    df["has_cover_image"] = df["coverImage_medium"].notna() if "coverImage_medium" in df.columns else False
    df["has_banner_image"] = df["bannerImage"].notna() if "bannerImage" in df.columns else False
    df["has_trailer"] = df["trailer_id"].notna() if "trailer_id" in df.columns else False
    return df


def main() -> None:
    args = _parse_args()
    input_csv = Path(args.input_csv)
    mapping_json = Path(args.mapping_json)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not mapping_json.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {mapping_json}")

    mapping = _read_mapping(mapping_json)
    df = pd.read_csv(input_csv)
    df = df.rename(columns=mapping)

    # Ensure required columns exist before transformation.
    for col in PROCESSED_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    _coerce_numeric(
        df,
        [
            "id",
            "seasonYear",
            "episodes",
            "duration",
            "averageScore",
            "meanScore",
            "popularity",
            "favourites",
            "trending",
            "startDate_year",
            "startDate_month",
            "startDate_day",
        ],
    )
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    _apply_interim_like_imputation(df)
    _apply_processed_like_rules(df)

    processed_out = _select_columns(df, PROCESSED_REQUIRED_COLUMNS + [
        "release_year",
        "release_quarter",
        "release_quarter_key",
        "popularity_quarter_pct",
        "popularity_quarter_bucket",
        "split_pre_release_effective",
        "is_model_split",
    ])

    multimodal_base = _with_multimodal_flags(df)
    multimodal_out = _select_columns(multimodal_base, MULTIMODAL_CONTRACT_COLUMNS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    processed_path = OUT_DIR / f"{args.output_prefix}_processed_contract_{ts}.csv"
    multimodal_path = OUT_DIR / f"{args.output_prefix}_multimodal_contract_{ts}.csv"
    summary_path = OUT_DIR / f"{args.output_prefix}_transform_summary_{ts}.json"

    processed_out.to_csv(processed_path, index=False)
    multimodal_out.to_csv(multimodal_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": input_csv.as_posix(),
        "mapping_json": mapping_json.as_posix(),
        "input_rows": int(len(df)),
        "processed_contract_rows": int(len(processed_out)),
        "multimodal_contract_rows": int(len(multimodal_out)),
        "processed_contract_columns": int(len(processed_out.columns)),
        "multimodal_contract_columns": int(len(multimodal_out.columns)),
        "output_files": {
            "processed_contract_csv": processed_path.as_posix(),
            "multimodal_contract_csv": multimodal_path.as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {processed_path}")
    print(f"Wrote {multimodal_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
