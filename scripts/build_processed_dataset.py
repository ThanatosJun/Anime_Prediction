"""
Build processed dataset with explicit outlier handling rules.

Input:
- latest `data/interim/anilist_anime_data_interim_*.csv`

Outputs:
- `data/processed/anilist_anime_data_processed_v1.csv`
- `data/processed/anilist_anime_data_processed_v1_meta.json`
- `data/eda/outlier_handling_summary.md`
- `data/eda/outlier_handling_summary.json`
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
EDA_DIR = Path("data/eda")

PROCESSED_CSV = PROCESSED_DIR / "anilist_anime_data_processed_v1.csv"
PROCESSED_META = PROCESSED_DIR / "anilist_anime_data_processed_v1_meta.json"
OUTLIER_SUMMARY_JSON = EDA_DIR / "outlier_handling_summary.json"
OUTLIER_SUMMARY_MD = EDA_DIR / "outlier_handling_summary.md"
RULE_VERSION = "decision_eda_v1"

LOWER_BOUND_COLUMNS = ["episodes", "duration", "averageScore", "meanScore", "popularity", "favourites", "trending"]
CLIP_COLUMNS = {
    "episodes": (0.01, 0.99),
    "duration": (0.01, 0.99),
    "averageScore": (0.005, 0.995),
    "meanScore": (0.005, 0.995),
    "popularity": (0.01, 0.99),
    "favourites": (0.01, 0.99),
    "trending": (0.01, 0.95),
}


def _latest_interim_csv() -> Path:
    candidates = sorted(INTERIM_DIR.glob("anilist_anime_data_interim_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No interim CSV found. Run scripts/build_interim_dataset.py first."
        )
    return candidates[-1]


def _enforce_non_negative(df: pd.DataFrame) -> dict:
    stats = {}
    for col in LOWER_BOUND_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        negative_count = int((series < 0).sum())
        df[col] = series.clip(lower=0)
        stats[col] = {"negative_to_zero_count": negative_count}
    return stats


def _clip_by_percentile(df: pd.DataFrame) -> dict:
    stats = {}
    for col, (q_low, q_high) in CLIP_COLUMNS.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue
        lower = float(clean.quantile(q_low))
        upper = float(clean.quantile(q_high))
        clipped = series.clip(lower=lower, upper=upper)
        changed_count = int((series != clipped).fillna(False).sum())
        df[col] = clipped
        stats[col] = {
            "lower_quantile": q_low,
            "upper_quantile": q_high,
            "lower_bound": lower,
            "upper_bound": upper,
            "clipped_count": changed_count,
        }
    return stats


def _write_summary(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    OUTLIER_SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Outlier Handling Summary",
        "",
        f"- Rule version: `{summary['rule_version']}`",
        f"- Input file: `{summary['input_file']}`",
        f"- Output rows: `{summary['row_count']}`",
        f"- Output columns: `{summary['column_count']}`",
        "",
        "## Non-negative Enforcement",
        "",
    ]

    if summary["non_negative"]:
        for col, values in summary["non_negative"].items():
            lines.append(
                f"- `{col}`: negative_to_zero={values['negative_to_zero_count']}"
            )
    else:
        lines.append("- No applicable columns found.")

    lines.extend(["", "## Percentile Clipping", ""])
    if summary["percentile_clipping"]:
        for col, values in summary["percentile_clipping"].items():
            lines.append(
                f"- `{col}`: [{values['lower_bound']:.4f}, {values['upper_bound']:.4f}], "
                f"clipped={values['clipped_count']}"
            )
    else:
        lines.append("- No clipping was applied.")

    OUTLIER_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    input_file = _latest_interim_csv()
    df = pd.read_csv(input_file)

    non_negative_stats = _enforce_non_negative(df)
    clipping_stats = _clip_by_percentile(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)

    meta = {
        "rule_version": RULE_VERSION,
        "input_file": str(input_file.as_posix()),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "clip_config": CLIP_COLUMNS,
        "non_negative": non_negative_stats,
        "percentile_clipping": clipping_stats,
    }
    PROCESSED_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(meta)

    print(f"Wrote {PROCESSED_CSV}")
    print(f"Wrote {PROCESSED_META}")
    print(f"Wrote {OUTLIER_SUMMARY_JSON}")
    print(f"Wrote {OUTLIER_SUMMARY_MD}")


if __name__ == "__main__":
    main()
