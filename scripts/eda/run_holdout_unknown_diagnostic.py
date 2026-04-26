"""
Generate diagnostics for holdout_unknown samples.

Outputs:
- data/eda/holdout_unknown_diagnostic.json
- data/eda/holdout_unknown_diagnostic.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROCESSED_CSV = Path("data/processed/anilist_anime_data_processed_v1.csv")
EDA_DIR = Path("data/eda")
OUTPUT_JSON = EDA_DIR / "holdout_unknown_diagnostic.json"
OUTPUT_MD = EDA_DIR / "holdout_unknown_diagnostic.md"

COMPARE_COLUMNS = ["popularity", "averageScore", "episodes", "duration", "favourites", "trending"]


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _median_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def build_summary(df: pd.DataFrame) -> dict:
    holdout = df[df["split_pre_release_effective"] == "holdout_unknown"].copy()
    model_data = df[df["is_model_split"] == True].copy()

    split_counts = (
        df["split_pre_release_effective"].value_counts(dropna=False).rename_axis("split").reset_index(name="count")
    )
    split_counts_list = [{"split": str(row["split"]), "count": int(row["count"])} for _, row in split_counts.iterrows()]

    missing_focus = {}
    for col in ["release_year", "release_quarter", "release_quarter_key", "seasonYear", "startDate_year", "startDate_month"]:
        if col in holdout.columns:
            missing_focus[col] = float(holdout[col].isna().mean())

    distribution_gap = {}
    for col in COMPARE_COLUMNS:
        if col not in df.columns:
            continue
        holdout_mean = _mean_or_none(holdout[col])
        model_mean = _mean_or_none(model_data[col])
        holdout_median = _median_or_none(holdout[col])
        model_median = _median_or_none(model_data[col])
        distribution_gap[col] = {
            "holdout_mean": holdout_mean,
            "model_mean": model_mean,
            "mean_gap": None if holdout_mean is None or model_mean is None else holdout_mean - model_mean,
            "holdout_median": holdout_median,
            "model_median": model_median,
            "median_gap": None if holdout_median is None or model_median is None else holdout_median - model_median,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": int(len(df)),
        "holdout_unknown_rows": int(len(holdout)),
        "holdout_unknown_ratio": float(len(holdout) / max(len(df), 1)),
        "split_counts": split_counts_list,
        "temporal_missing_focus": missing_focus,
        "distribution_gap_vs_model_split": distribution_gap,
        "policy_note": "holdout_unknown samples are excluded from model train/val/test.",
    }


def write_outputs(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Holdout Unknown Diagnostic",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Total rows: `{summary['total_rows']}`",
        f"- holdout_unknown rows: `{summary['holdout_unknown_rows']}` ({summary['holdout_unknown_ratio']:.2%})",
        f"- Policy: `{summary['policy_note']}`",
        "",
        "## Effective Split Counts",
        "",
    ]
    for item in summary["split_counts"]:
        lines.append(f"- `{item['split']}`: {item['count']}")

    lines.extend(["", "## Temporal Missing Focus (holdout_unknown only)", ""])
    for col, ratio in summary["temporal_missing_focus"].items():
        lines.append(f"- `{col}` missing ratio: `{ratio:.2%}`")

    lines.extend(["", "## Distribution Gap vs Model Splits", ""])
    for col, info in summary["distribution_gap_vs_model_split"].items():
        lines.append(
            f"- `{col}`: mean_gap={info['mean_gap']}, median_gap={info['median_gap']}"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError("Processed CSV not found. Run scripts/build_processed_dataset.py first.")
    df = pd.read_csv(PROCESSED_CSV)
    if "split_pre_release_effective" not in df.columns or "is_model_split" not in df.columns:
        raise ValueError("Processed dataset is missing split columns. Rebuild processed dataset.")

    summary = build_summary(df)
    write_outputs(summary)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
