"""Build a ranked markdown/CSV report from text-branch quality comparison results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


LOWER_IS_BETTER_SUFFIXES = ("_MAE", "_RMSE")
HIGHER_IS_BETTER_SUFFIXES = ("_Spearman",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ranked quality report for text model experiments.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default="text_branch_quality_compare.csv",
        help="Input comparison CSV filename under report_dir.",
    )
    parser.add_argument(
        "--ranked-csv-name",
        type=str,
        default="text_branch_quality_ranked.csv",
        help="Output ranked CSV filename under report_dir.",
    )
    parser.add_argument(
        "--markdown-name",
        type=str,
        default="text_branch_quality_ranking.md",
        help="Output markdown summary filename under report_dir.",
    )
    return parser.parse_args()


def _metric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col.endswith(LOWER_IS_BETTER_SUFFIXES) or col.endswith(HIGHER_IS_BETTER_SUFFIXES):
            cols.append(col)
    return cols


def _rank_metric(df: pd.DataFrame, col: str) -> pd.Series:
    ascending = col.endswith(LOWER_IS_BETTER_SUFFIXES)
    return df[col].rank(method="min", ascending=ascending)


def _build_markdown(df: pd.DataFrame, metric_cols: List[str]) -> str:
    lines = [
        "# Text Branch Quality Ranking",
        "",
        "Ranking policy:",
        "- Selection rank uses validation metrics only (lower MAE/RMSE is better, higher Spearman is better).",
        "- Test rank is reported separately as a holdout view.",
        "",
        "## Leaderboard",
        "",
        "| Rank | Experiment | Model Key | Val Rank Avg | Test Rank Avg | popularity Test Spearman | meanScore Test Spearman | popularity Test RMSE | meanScore Test RMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for idx, row in df.iterrows():
        lines.append(
            "| {rank} | {experiment} | {key} | {val_rank:.2f} | {test_rank:.2f} | {pop_spear:.4f} | {score_spear:.4f} | {pop_rmse:.4f} | {score_rmse:.4f} |".format(
                rank=int(idx + 1),
                experiment=row.get("experiment_name", ""),
                key=row.get("embedding_model_key", ""),
                val_rank=row.get("selection_rank_avg", 0.0),
                test_rank=row.get("test_rank_avg", 0.0),
                pop_spear=row.get("popularity_test_Spearman", 0.0),
                score_spear=row.get("meanScore_test_Spearman", 0.0),
                pop_rmse=row.get("popularity_test_RMSE", 0.0),
                score_rmse=row.get("meanScore_test_RMSE", 0.0),
            )
        )

    lines.extend([
        "",
        "## Metric Columns Used",
        "",
    ])
    for col in metric_cols:
        lines.append(f"- {col}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    compare_path = args.report_dir / args.comparison_csv
    if not compare_path.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {compare_path}")

    df = pd.read_csv(compare_path)
    if df.empty:
        raise ValueError(f"Comparison CSV is empty: {compare_path}")

    metric_cols = _metric_columns(df)
    if not metric_cols:
        raise ValueError("No metric columns found to rank")

    val_metric_cols = [c for c in metric_cols if "_val_" in c]
    test_metric_cols = [c for c in metric_cols if "_test_" in c]

    ranked_df = df.copy()
    for col in metric_cols:
        ranked_df[f"rank__{col}"] = _rank_metric(ranked_df, col)

    ranked_df["selection_rank_avg"] = ranked_df[[f"rank__{c}" for c in val_metric_cols]].mean(axis=1)
    ranked_df["test_rank_avg"] = ranked_df[[f"rank__{c}" for c in test_metric_cols]].mean(axis=1)
    ranked_df = ranked_df.sort_values(
        by=["selection_rank_avg", "test_rank_avg"],
        ascending=[True, True],
    ).reset_index(drop=True)
    ranked_df.insert(0, "overall_rank", ranked_df.index + 1)

    ranked_csv_path = args.report_dir / args.ranked_csv_name
    ranked_df.to_csv(ranked_csv_path, index=False, encoding="utf-8")

    markdown_path = args.report_dir / args.markdown_name
    markdown_path.write_text(_build_markdown(ranked_df, metric_cols), encoding="utf-8")

    print(f"Saved ranked CSV: {ranked_csv_path.as_posix()}")
    print(f"Saved markdown report: {markdown_path.as_posix()}")


if __name__ == "__main__":
    main()
