"""
Run lightweight baseline EDA for the AniList dataset.

Outputs:
- data/eda/baseline_eda_summary.json
- data/eda/baseline_eda_summary.md
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
RAW_XLSX = RAW_DIR / "anilist_anime_data_complete.xlsx"

SUMMARY_JSON = EDA_DIR / "baseline_eda_summary.json"
SUMMARY_MD = EDA_DIR / "baseline_eda_summary.md"

KEY_NUMERIC_COLUMNS = ["popularity", "averageScore", "meanScore", "episodes", "duration"]


def _load_raw_dataset() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    if RAW_XLSX.exists():
        return pd.read_excel(RAW_XLSX, sheet_name="Sheet1")
    raise FileNotFoundError("No supported raw dataset found in data/raw.")


def _to_builtin(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _numeric_distribution(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
        }

    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "q1": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q3": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def _iqr_outlier_bounds(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"lower": None, "upper": None, "outlier_count": 0, "outlier_ratio": 0.0}

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (clean < lower) | (clean > upper)

    return {
        "lower": float(lower),
        "upper": float(upper),
        "outlier_count": int(mask.sum()),
        "outlier_ratio": float(mask.mean()),
    }


def build_summary(df: pd.DataFrame) -> dict:
    working = df.copy()
    for col in KEY_NUMERIC_COLUMNS:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    rows = len(working)
    cols = len(working.columns)

    duplicate_id_count = 0
    if "id" in working.columns:
        duplicate_id_count = int(working["id"].duplicated().sum())

    missing_rates = (
        working.isna().mean().sort_values(ascending=False).head(20).to_dict()
        if rows > 0
        else {}
    )

    numeric_summary = {
        col: _numeric_distribution(working[col])
        for col in KEY_NUMERIC_COLUMNS
        if col in working.columns
    }

    outlier_bounds = {
        col: _iqr_outlier_bounds(working[col])
        for col in ["popularity", "averageScore", "episodes", "duration"]
        if col in working.columns
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": rows,
        "column_count": cols,
        "id_duplicate_count": duplicate_id_count,
        "missing_rate_top20": {
            key: float(value) for key, value in missing_rates.items()
        },
        "numeric_summary": numeric_summary,
        "outlier_bounds_iqr": outlier_bounds,
    }


def write_outputs(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_to_builtin)

    lines = [
        "# Baseline EDA Summary",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Rows: `{summary['row_count']}`",
        f"- Columns: `{summary['column_count']}`",
        f"- Duplicate `id` count: `{summary['id_duplicate_count']}`",
        "",
        "## Missing Rate (Top 20)",
        "",
    ]

    if summary["missing_rate_top20"]:
        for col, ratio in summary["missing_rate_top20"].items():
            lines.append(f"- `{col}`: `{ratio:.4%}`")
    else:
        lines.append("- No missing-rate data available.")

    lines.extend(["", "## Key Numeric Distribution", ""])
    for col, stats in summary["numeric_summary"].items():
        lines.append(
            f"- `{col}`: count={stats['count']}, "
            f"mean={stats['mean']}, q1={stats['q1']}, median={stats['median']}, q3={stats['q3']}"
        )

    lines.extend(["", "## IQR Outlier Bounds", ""])
    for col, bounds in summary["outlier_bounds_iqr"].items():
        lines.append(
            f"- `{col}`: lower={bounds['lower']}, upper={bounds['upper']}, "
            f"outliers={bounds['outlier_count']} ({bounds['outlier_ratio']:.4%})"
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = _load_raw_dataset()
    summary = build_summary(df)
    write_outputs(summary)
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
