"""
Generate missing-value reports for full data vs model-split-only data.

Outputs:
- data/eda/missing_value_report.json
- data/eda/missing_value_report.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROCESSED_CSV = Path("data/processed/anilist_anime_data_processed_v1.csv")
MULTIMODAL_CSV = Path("data/processed/anilist_anime_multimodal_input_v1.csv")
EDA_DIR = Path("data/eda")
OUT_JSON = EDA_DIR / "missing_value_report.json"
OUT_MD = EDA_DIR / "missing_value_report.md"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path.as_posix()}")
    return pd.read_csv(path)


def _missing_table(df: pd.DataFrame) -> list[dict]:
    miss_rate = df.isna().mean().sort_values(ascending=False)
    miss_count = df.isna().sum()
    rows = []
    for col, rate in miss_rate.items():
        count = int(miss_count[col])
        if count == 0:
            continue
        rows.append(
            {
                "column": col,
                "missing_count": count,
                "missing_rate": float(rate),
            }
        )
    return rows


def _subset_without_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "is_model_split" in df.columns:
        return df[df["is_model_split"] == True].copy(), "is_model_split == True"
    if "split_pre_release_effective" in df.columns:
        return (
            df[df["split_pre_release_effective"].isin(["train", "val", "test"])].copy(),
            "split_pre_release_effective in {train,val,test}",
        )
    return df.copy(), "no split column found (same as full data)"


def _build_dataset_summary(name: str, df: pd.DataFrame) -> dict:
    subset_df, subset_rule = _subset_without_holdout(df)
    return {
        "dataset_name": name,
        "full_data": {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "missing_columns": _missing_table(df),
        },
        "without_holdout_unknown": {
            "subset_rule": subset_rule,
            "row_count": int(len(subset_df)),
            "column_count": int(len(subset_df.columns)),
            "missing_columns": _missing_table(subset_df),
        },
    }


def _write_markdown(summary: dict) -> None:
    lines = [
        "# Missing Value Report",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        "",
    ]

    for dataset in summary["datasets"]:
        lines.extend(
            [
                f"## {dataset['dataset_name']}",
                "",
                "### Full Data",
                f"- Rows: `{dataset['full_data']['row_count']}`",
                f"- Columns: `{dataset['full_data']['column_count']}`",
                "",
            ]
        )
        if dataset["full_data"]["missing_columns"]:
            for row in dataset["full_data"]["missing_columns"]:
                lines.append(
                    f"- `{row['column']}`: count=`{row['missing_count']}`, rate=`{row['missing_rate']:.6f}`"
                )
        else:
            lines.append("- No missing values.")

        lines.extend(
            [
                "",
                "### Without `holdout_unknown`",
                f"- Subset rule: `{dataset['without_holdout_unknown']['subset_rule']}`",
                f"- Rows: `{dataset['without_holdout_unknown']['row_count']}`",
                f"- Columns: `{dataset['without_holdout_unknown']['column_count']}`",
                "",
            ]
        )
        if dataset["without_holdout_unknown"]["missing_columns"]:
            for row in dataset["without_holdout_unknown"]["missing_columns"]:
                lines.append(
                    f"- `{row['column']}`: count=`{row['missing_count']}`, rate=`{row['missing_rate']:.6f}`"
                )
        else:
            lines.append("- No missing values.")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    processed_df = _load_csv(PROCESSED_CSV)
    multimodal_df = _load_csv(MULTIMODAL_CSV)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": [
            _build_dataset_summary("processed_v1", processed_df),
            _build_dataset_summary("multimodal_input_v1", multimodal_df),
        ],
    }

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary)

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
