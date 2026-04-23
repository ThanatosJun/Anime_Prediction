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
RULE_VERSION = "decision_eda_v2_relation_features"

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
    "is_sequel",
    "has_sequel",
    "prequel_count",
    "prequel_popularity_mean",
    "prequel_meanScore_mean",
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
    "prequel_count",
    "prequel_popularity_mean",
    "prequel_meanScore_mean",
]

MISSING_RULES = {
    "episodes": {"method": "format_median_then_global_median"},
    "duration": {"method": "format_median_then_global_median"},
    "averageScore": {"method": "meanScore_then_global_median"},
    "seasonYear": {"method": "startDate_year"},
    "title_english": {"method": "title_romaji"},
    "prequel_count": {"method": "fill_zero_when_no_prequel_match"},
    "prequel_popularity_mean": {"method": "fill_zero_when_no_prequel_match"},
    "prequel_meanScore_mean": {"method": "fill_zero_when_no_prequel_match"},
    "is_sequel": {"method": "fill_false_when_missing"},
    "has_sequel": {"method": "fill_false_when_missing"},
}


def load_raw_dataset() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError("No supported raw dataset found in data/raw (expected PKL or CSV).")


def add_relation_features(raw_df: pd.DataFrame, interim_df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in raw_df.columns or "relations" not in raw_df.columns:
        return interim_df

    popularity_lookup = (
        pd.Series(pd.to_numeric(raw_df.get("popularity"), errors="coerce").values, index=raw_df["id"])
        .groupby(level=0)
        .first()
    )
    score_lookup = (
        pd.Series(pd.to_numeric(raw_df.get("meanScore"), errors="coerce").values, index=raw_df["id"])
        .groupby(level=0)
        .first()
    )

    def _parse_relations(value: object) -> list[dict]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, str):
            value = value.strip()
            if not value or value in {"[]", "nan", "None"}:
                return []
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
            except Exception:
                return []
        return []

    rows = []
    for _, row in raw_df[["id", "relations"]].iterrows():
        relation_items = _parse_relations(row["relations"])
        prequel_ids: list[int] = []
        has_sequel = False

        for rel in relation_items:
            relation_type = str(rel.get("relationType", "")).upper()
            node = rel.get("node") if isinstance(rel.get("node"), dict) else {}
            node_type = str(node.get("type", "")).upper()
            node_id = node.get("id")
            if relation_type == "SEQUEL":
                has_sequel = True
            if relation_type == "PREQUEL" and node_type == "ANIME" and node_id is not None:
                try:
                    prequel_ids.append(int(node_id))
                except Exception:
                    continue

        prequel_pop_values = [float(popularity_lookup.get(pid)) for pid in prequel_ids if pd.notna(popularity_lookup.get(pid))]
        prequel_score_values = [float(score_lookup.get(pid)) for pid in prequel_ids if pd.notna(score_lookup.get(pid))]
        rows.append(
            {
                "id": row["id"],
                "is_sequel": bool(len(prequel_ids) > 0),
                "has_sequel": bool(has_sequel),
                "prequel_count": int(len(prequel_ids)),
                "prequel_popularity_mean": float(sum(prequel_pop_values) / len(prequel_pop_values))
                if prequel_pop_values
                else 0.0,
                "prequel_meanScore_mean": float(sum(prequel_score_values) / len(prequel_score_values))
                if prequel_score_values
                else 0.0,
            }
        )

    relation_features = pd.DataFrame(rows)
    return interim_df.merge(relation_features, on="id", how="left")


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

    if "title_english" in df.columns and "title_romaji" in df.columns:
        df["title_english"] = df["title_english"].fillna(df["title_romaji"])

    for col in ["prequel_count", "prequel_popularity_mean", "prequel_meanScore_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["is_sequel", "has_sequel"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

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
    interim_df = add_relation_features(raw_df, interim_df)
    interim_df = enforce_dtypes(interim_df)
    interim_df, removed_duplicates = deduplicate_rows(interim_df)
    interim_df = impute_missing_values(interim_df)
    interim_df = add_release_date(interim_df)

    metadata = {
        "rule_version": RULE_VERSION,
        "applied_missing_rules": MISSING_RULES,
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
