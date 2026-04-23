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
