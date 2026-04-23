"""
Generate column lineage report across raw/interim/processed stages.

Outputs:
- data/eda/column_lineage_summary.json
- data/eda/column_lineage_summary.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_PICKLE = Path("data/raw/anilist_anime_data_complete.pkl")
RAW_CSV = Path("data/raw/anilist_anime_data_complete.csv")
INTERIM_DIR = Path("data/interim")
PROCESSED_CSV = Path("data/processed/anilist_anime_data_processed_v1.csv")
EDA_DIR = Path("data/eda")

OUTPUT_JSON = EDA_DIR / "column_lineage_summary.json"
OUTPUT_MD = EDA_DIR / "column_lineage_summary.md"


def _load_raw() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError("Raw dataset not found.")


def _load_latest_interim() -> pd.DataFrame:
    candidates = sorted(INTERIM_DIR.glob("anilist_anime_data_interim_*.csv"))
    if not candidates:
        raise FileNotFoundError("No interim dataset found.")
    return pd.read_csv(candidates[-1])


def _load_processed() -> pd.DataFrame:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError("Processed dataset not found.")
    return pd.read_csv(PROCESSED_CSV)


def _stage_diff(src: set[str], dst: set[str]) -> dict:
    return {
        "kept": sorted(list(src & dst)),
        "dropped": sorted(list(src - dst)),
        "added": sorted(list(dst - src)),
    }


def _reason_for_raw_drop(column: str) -> str:
    reason_map = {
        # Deeply nested / high-cardinality structures (moved to later multimodal/retrieval stage)
        "characters": "high-cardinality nested structure; deferred to dedicated retrieval/graph feature stage",
        "staff": "high-cardinality nested structure; deferred to dedicated retrieval/graph feature stage",
        "relations": "graph-style relation payload; handled in retrieval augmentation stage instead of baseline tabular stage",
        "recommendations": "nested recommendation graph payload; excluded from baseline tabular preprocessing",
        "reviews": "nested text payload with variable quality/length; deferred to dedicated NLP stage",
        "airingSchedule": "nested schedule nodes; not required for current pre-release baseline target engineering",
        "streamingEpisodes": "nested streaming payload; not stable for pre-release baseline feature contract",
        "externalLinks": "nested external metadata; excluded to keep baseline feature contract compact",
        "rankings": "nested ranking history; may introduce post-release leakage and high variance",
        "tags": "high-cardinality nested tags; deferred to dedicated text/tag embedding pipeline",
        # Near-duplicate or redundant title fields
        "title_userPreferred": "redundant with retained title fields and may vary by locale/user preference",
        "synonyms": "high-variance synonym list; deferred to dedicated text normalization stage",
        # Potential leakage / post-release dynamics
        "nextAiringEpisode": "post-schedule dynamic field; excluded to avoid temporal inconsistency/leakage risk",
        "trailer_id": "retained as availability metric in EDA but excluded from baseline tabular contract to avoid sparse-noisy key field",
        "trailer_site": "sparse categorical trailer metadata; deferred to multimodal source availability analysis",
        "trailer_thumbnail": "media URL field; excluded from baseline tabular contract",
        "siteUrl": "identifier-style URL field; non-semantic for baseline model input",
        "updatedAt": "platform update timestamp; can encode non-stationary platform behavior",
        # Extremely sparse or non-anime oriented
        "chapters": "manga-oriented field; 100% missing for this anime-focused pipeline",
        "volumes": "manga-oriented field; 100% missing for this anime-focused pipeline",
        "hashtag": "high missingness and unstable social metadata",
        "bannerImage": "auxiliary media field; baseline uses tabular contract and defers image handling to dedicated stage",
        "description": "kept for RQ readiness analysis but excluded from baseline tabular feature set (reserved for text encoder stage)",
        # Retained elsewhere in processed transforms or not needed now
        "coverImage_extraLarge": "raw media URL variant; excluded from baseline tabular contract",
        "coverImage_large": "raw media URL variant; excluded from baseline tabular contract",
        "coverImage_medium": "tracked in RQ coverage analysis but not used as direct tabular scalar feature",
        "coverImage_color": "auxiliary color metadata; low impact baseline feature and missingness present",
        # Distribution meta payloads
        "stats_scoreDistribution": "aggregated platform distribution payload; excluded from row-level baseline contract",
        "stats_statusDistribution": "aggregated platform distribution payload; excluded from row-level baseline contract",
        # IDs / flags not needed for baseline
        "idMal": "external ID mapping field; excluded from baseline predictive feature set",
        "isFavourite": "user-level interaction flag; unstable and potentially leakage-prone",
        "isLicensed": "platform/legal flag not required in current baseline contract",
        "isLocked": "platform operational flag; not semantically meaningful for target task",
        "seasonInt": "redundant encoded season field; season + seasonYear retained",
        "source": "retained in interim/processed",
        "endDate_year": "post-release endpoint detail not required for pre-release feature contract",
        "endDate_month": "post-release endpoint detail not required for pre-release feature contract",
        "endDate_day": "post-release endpoint detail not required for pre-release feature contract",
    }
    return reason_map.get(column, "excluded to keep baseline tabular contract compact and reproducible")


def build_summary(raw_df: pd.DataFrame, interim_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    raw_cols = set(raw_df.columns.tolist())
    interim_cols = set(interim_df.columns.tolist())
    processed_cols = set(processed_df.columns.tolist())

    raw_to_interim = _stage_diff(raw_cols, interim_cols)
    interim_to_processed = _stage_diff(interim_cols, processed_cols)
    raw_to_processed = _stage_diff(raw_cols, processed_cols)

    known_derived_fields = {
        "release_date": "derived in interim (`add_release_date`)",
        "release_year": "derived in processed (`_derive_release_quarter`)",
        "release_quarter": "derived in processed (`_derive_release_quarter`)",
        "release_quarter_key": "derived in processed (`_derive_release_quarter`)",
        "popularity_quarter_pct": "derived in processed (`_add_popularity_quarter_target`)",
        "popularity_quarter_bucket": "derived in processed (`_add_popularity_quarter_target`)",
        "split_pre_release": "derived in processed (`_apply_pre_release_temporal_split`)",
        "split_pre_release_effective": "derived in processed (`_apply_unknown_split_policy`)",
        "is_model_split": "derived in processed (`_apply_unknown_split_policy`)",
    }
    raw_drop_reasons = {col: _reason_for_raw_drop(col) for col in raw_to_interim["dropped"]}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_column_counts": {
            "raw": int(len(raw_cols)),
            "interim": int(len(interim_cols)),
            "processed": int(len(processed_cols)),
        },
        "raw_to_interim": raw_to_interim,
        "interim_to_processed": interim_to_processed,
        "raw_to_processed": raw_to_processed,
        "known_derived_fields": known_derived_fields,
        "raw_drop_reasons": raw_drop_reasons,
    }


def write_outputs(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = summary["stage_column_counts"]
    lines = [
        "# Column Lineage Summary",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Column counts: raw=`{counts['raw']}`, interim=`{counts['interim']}`, processed=`{counts['processed']}`",
        "",
        "## Raw -> Interim",
        "",
        f"- Kept columns: `{len(summary['raw_to_interim']['kept'])}`",
        f"- Dropped columns: `{len(summary['raw_to_interim']['dropped'])}`",
        f"- Added columns: `{len(summary['raw_to_interim']['added'])}`",
        "",
        "### Added in Interim",
    ]
    for col in summary["raw_to_interim"]["added"]:
        note = summary["known_derived_fields"].get(col, "derived")
        lines.append(f"- `{col}`: {note}")

    lines.extend(
        [
            "",
            "## Interim -> Processed",
            "",
            f"- Kept columns: `{len(summary['interim_to_processed']['kept'])}`",
            f"- Dropped columns: `{len(summary['interim_to_processed']['dropped'])}`",
            f"- Added columns: `{len(summary['interim_to_processed']['added'])}`",
            "",
            "### Added in Processed",
        ]
    )
    for col in summary["interim_to_processed"]["added"]:
        note = summary["known_derived_fields"].get(col, "derived")
        lines.append(f"- `{col}`: {note}")

    lines.extend(["", "## Raw -> Processed Direct View", ""])
    lines.append(f"- Dropped from raw by final stage: `{len(summary['raw_to_processed']['dropped'])}`")
    lines.append(f"- Added by final stage: `{len(summary['raw_to_processed']['added'])}`")

    lines.extend(["", "## Raw -> Interim Drop Reasons", ""])
    for col in summary["raw_to_interim"]["dropped"]:
        lines.append(f"- `{col}`: {summary['raw_drop_reasons'].get(col)}")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_df = _load_raw()
    interim_df = _load_latest_interim()
    processed_df = _load_processed()
    summary = build_summary(raw_df, interim_df, processed_df)
    write_outputs(summary)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
