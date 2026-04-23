"""
Generate decision-oriented EDA outputs for data cleaning and outlier policy.

Outputs:
- data/eda/decision_eda_summary.json
- data/eda/decision_eda_summary.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
EDA_DIR = Path("data/eda")

RAW_PICKLE = RAW_DIR / "anilist_anime_data_complete.pkl"
RAW_CSV = RAW_DIR / "anilist_anime_data_complete.csv"

SUMMARY_JSON = EDA_DIR / "decision_eda_summary.json"
SUMMARY_MD = EDA_DIR / "decision_eda_summary.md"

TARGET_COLUMNS = ["popularity", "averageScore"]
ANALYSIS_NUMERIC_COLUMNS = [
    "episodes",
    "duration",
    "meanScore",
    "favourites",
    "trending",
    "seasonYear",
]
OUTLIER_COLUMNS = ["episodes", "duration", "averageScore", "meanScore", "popularity", "favourites", "trending"]


def _load_raw_dataset() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError("No supported raw dataset found in data/raw (expected PKL or CSV).")


def _missing_strategy(missing_ratio: float) -> str:
    if missing_ratio >= 0.90:
        return "drop"
    if missing_ratio >= 0.20:
        return "fill"
    return "keep"


def _outlier_strategy(outlier_ratio: float) -> str:
    if outlier_ratio >= 0.10:
        return "clip_p1_p99"
    if outlier_ratio >= 0.03:
        return "winsorize_p1_p99"
    return "retain"


def _corr_to_targets(df: pd.DataFrame, column: str) -> dict:
    values = {}
    for target in TARGET_COLUMNS:
        if target in df.columns and column in df.columns:
            corr = df[column].corr(df[target])
            values[target] = None if pd.isna(corr) else float(corr)
    return values


def _iqr_profile(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_ratio": 0.0,
            "recommended_strategy": "retain",
        }

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (clean < lower) | (clean > upper)
    outlier_ratio = float(outlier_mask.mean())

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outlier_ratio": outlier_ratio,
        "recommended_strategy": _outlier_strategy(outlier_ratio),
    }


def build_decision_summary(df: pd.DataFrame) -> dict:
    working = df.copy()
    numeric_columns = sorted(
        {
            col
            for col in (ANALYSIS_NUMERIC_COLUMNS + TARGET_COLUMNS + OUTLIER_COLUMNS)
            if col in working.columns
        }
    )
    for col in numeric_columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    missing_policy = {}
    for col in working.columns:
        missing_ratio = float(working[col].isna().mean())
        missing_policy[col] = {
            "missing_ratio": missing_ratio,
            "recommended_strategy": _missing_strategy(missing_ratio),
        }

    grouped_impact = {}
    if "format" in working.columns and "popularity" in working.columns:
        format_agg = (
            working.groupby("format", dropna=False)["popularity"]
            .agg(["count", "median", "mean"])
            .sort_values("count", ascending=False)
            .head(10)
        )
        grouped_impact["format_to_popularity_top10"] = [
            {
                "format": None if pd.isna(idx) else str(idx),
                "count": int(row["count"]),
                "median_popularity": float(row["median"]),
                "mean_popularity": float(row["mean"]),
            }
            for idx, row in format_agg.iterrows()
        ]

    correlation_profile = {}
    for col in ANALYSIS_NUMERIC_COLUMNS:
        if col in working.columns:
            correlation_profile[col] = _corr_to_targets(working, col)

    outlier_policy = {}
    for col in OUTLIER_COLUMNS:
        if col in working.columns:
            outlier_policy[col] = _iqr_profile(working[col])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(working)),
        "column_count": int(len(working.columns)),
        "decision_rules": {
            "missing_policy": missing_policy,
            "outlier_policy": outlier_policy,
            "correlation_profile": correlation_profile,
            "grouped_impact": grouped_impact,
        },
    }


def write_outputs(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    missing_policy = summary["decision_rules"]["missing_policy"]
    outlier_policy = summary["decision_rules"]["outlier_policy"]

    lines = [
        "# Decision EDA Summary",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Columns: `{summary['column_count']}`",
        "",
        "## Missing Value Policy Recommendations",
        "",
    ]

    prioritized_missing = sorted(
        missing_policy.items(),
        key=lambda item: item[1]["missing_ratio"],
        reverse=True,
    )[:20]
    for col, info in prioritized_missing:
        lines.append(
            f"- `{col}`: missing={info['missing_ratio']:.4%}, recommend=`{info['recommended_strategy']}`"
        )

    lines.extend(["", "## Outlier Policy Recommendations", ""])
    for col, info in outlier_policy.items():
        lines.append(
            f"- `{col}`: outlier_ratio={info['outlier_ratio']:.4%}, "
            f"bounds=[{info['lower_bound']}, {info['upper_bound']}], "
            f"recommend=`{info['recommended_strategy']}`"
        )

    lines.extend(["", "## Correlation Profile (to targets)", ""])
    for col, corr in summary["decision_rules"]["correlation_profile"].items():
        lines.append(
            f"- `{col}`: popularity={corr.get('popularity')}, averageScore={corr.get('averageScore')}"
        )

    grouped = summary["decision_rules"]["grouped_impact"].get("format_to_popularity_top10", [])
    lines.extend(["", "## Format Impact on Popularity (Top 10 by count)", ""])
    for item in grouped:
        lines.append(
            f"- `{item['format']}`: count={item['count']}, median={item['median_popularity']}, mean={item['mean_popularity']}"
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = _load_raw_dataset()
    summary = build_decision_summary(df)
    write_outputs(summary)
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
