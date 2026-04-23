"""
Generate research-question-oriented EDA outputs.

Outputs:
- data/eda/rq_eda_summary.json
- data/eda/rq_eda_summary.md
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

RQ_SUMMARY_JSON = EDA_DIR / "rq_eda_summary.json"
RQ_SUMMARY_MD = EDA_DIR / "rq_eda_summary.md"


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


def _missing_ratio(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    return float(df[column].isna().mean())


def _corr_safe(a: pd.Series, b: pd.Series) -> float | None:
    corr = pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"))
    return None if pd.isna(corr) else float(corr)


def build_summary(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    coverage = {
        "description_missing_ratio": _missing_ratio(raw_df, "description"),
        "coverImage_medium_missing_ratio": _missing_ratio(raw_df, "coverImage_medium"),
        "trailer_id_missing_ratio": _missing_ratio(raw_df, "trailer_id"),
        "studios_missing_ratio": _missing_ratio(processed_df, "studios"),
        "genres_missing_ratio": _missing_ratio(processed_df, "genres"),
    }

    snapshot_control = {}
    if "release_year" in processed_df.columns:
        snapshot_control["corr_release_year_vs_popularity_raw"] = _corr_safe(
            processed_df["release_year"], processed_df["popularity"]
        )
    if "release_year" in processed_df.columns and "popularity_quarter_pct" in processed_df.columns:
        snapshot_control["corr_release_year_vs_popularity_quarter_pct"] = _corr_safe(
            processed_df["release_year"], processed_df["popularity_quarter_pct"]
        )

    rq1_proxy = {
        "metadata_relation_coverage": {
            "studios_available_ratio": None
            if coverage["studios_missing_ratio"] is None
            else 1.0 - coverage["studios_missing_ratio"],
            "genres_available_ratio": None
            if coverage["genres_missing_ratio"] is None
            else 1.0 - coverage["genres_missing_ratio"],
        },
        "split_distribution": processed_df["split_pre_release_effective"].value_counts(dropna=False).to_dict()
        if "split_pre_release_effective" in processed_df.columns
        else {},
    }

    rq2_proxy = {
        "multimodal_source_coverage": {
            "text_description_available_ratio": None
            if coverage["description_missing_ratio"] is None
            else 1.0 - coverage["description_missing_ratio"],
            "image_cover_available_ratio": None
            if coverage["coverImage_medium_missing_ratio"] is None
            else 1.0 - coverage["coverImage_medium_missing_ratio"],
            "trailer_id_available_ratio": None
            if coverage["trailer_id_missing_ratio"] is None
            else 1.0 - coverage["trailer_id_missing_ratio"],
        }
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_raw": int(len(raw_df)),
        "rows_processed": int(len(processed_df)),
        "coverage": coverage,
        "snapshot_control": snapshot_control,
        "rq1_retrieval_proxy": rq1_proxy,
        "rq2_multimodal_proxy": rq2_proxy,
    }


def write_outputs(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    RQ_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RQ-oriented EDA Summary",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Raw rows: `{summary['rows_raw']}`",
        f"- Processed rows: `{summary['rows_processed']}`",
        "",
        "## Snapshot Control Evidence",
        "",
    ]
    sc = summary["snapshot_control"]
    lines.append(f"- Corr(release_year, popularity_raw): `{sc.get('corr_release_year_vs_popularity_raw')}`")
    lines.append(
        f"- Corr(release_year, popularity_quarter_pct): `{sc.get('corr_release_year_vs_popularity_quarter_pct')}`"
    )

    lines.extend(["", "## RQ1 Proxy (Retrieval/Metadata Readiness)", ""])
    rq1 = summary["rq1_retrieval_proxy"]
    md_cov = rq1["metadata_relation_coverage"]
    lines.append(f"- Studios available ratio: `{md_cov.get('studios_available_ratio')}`")
    lines.append(f"- Genres available ratio: `{md_cov.get('genres_available_ratio')}`")
    for split_name, count in rq1.get("split_distribution", {}).items():
        lines.append(f"- Split `{split_name}` rows: `{count}`")

    lines.extend(["", "## RQ2 Proxy (Multimodal Readiness)", ""])
    rq2 = summary["rq2_multimodal_proxy"]["multimodal_source_coverage"]
    lines.append(f"- Text description available ratio: `{rq2.get('text_description_available_ratio')}`")
    lines.append(f"- Image cover available ratio: `{rq2.get('image_cover_available_ratio')}`")
    lines.append(f"- Trailer id available ratio: `{rq2.get('trailer_id_available_ratio')}`")

    RQ_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_df = _load_raw()
    processed_df = _load_processed()
    summary = build_summary(raw_df, processed_df)
    write_outputs(summary)
    print(f"Wrote {RQ_SUMMARY_JSON}")
    print(f"Wrote {RQ_SUMMARY_MD}")


if __name__ == "__main__":
    main()
