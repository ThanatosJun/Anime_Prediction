"""Analyze MAL 2025 label sanity and external calibration diagnostics.

This script is intentionally read-only with respect to model artifacts. It uses
the prepared MAL 2025 contract and existing Run22 prediction CSVs to produce
paper-facing diagnostics for external-label alignment and scale mismatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


PREDICTION_SPECS = [
    {
        "exam": "pop_only",
        "variant": "no_yolo",
        "path": "data/external_transformed/run22_mal2025_popularity_local_ready_predictions.csv",
        "targets": ["popularity"],
    },
    {
        "exam": "dual",
        "variant": "no_yolo",
        "path": "data/external_transformed/run22_mal2025_dual_local_ready_predictions.csv",
        "targets": ["popularity", "meanScore"],
    },
    {
        "exam": "pop_only",
        "variant": "cover_yolo",
        "path": "data/external_transformed/run22_mal2025_popularity_local_ready_yolo_predictions.csv",
        "targets": ["popularity"],
    },
    {
        "exam": "dual",
        "variant": "cover_yolo",
        "path": "data/external_transformed/run22_mal2025_dual_local_ready_yolo_predictions.csv",
        "targets": ["popularity", "meanScore"],
    },
    {
        "exam": "pop_only",
        "variant": "cover_yolo_coverbanner_proxy",
        "path": "data/external_transformed/run22_mal2025_popularity_local_ready_yolo_coverbanner_predictions.csv",
        "targets": ["popularity"],
    },
    {
        "exam": "dual",
        "variant": "cover_yolo_coverbanner_proxy",
        "path": "data/external_transformed/run22_mal2025_dual_local_ready_yolo_coverbanner_predictions.csv",
        "targets": ["popularity", "meanScore"],
    },
]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)


def _slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _metric_rows_for_pair(
    df: pd.DataFrame,
    source: str,
    target: str,
    true_col: str,
    pred_col: str,
    transform: str,
) -> dict:
    work = df[[true_col, pred_col]].dropna().copy()
    if transform == "log1p":
        true_eval = np.log1p(work[true_col].clip(lower=0).to_numpy(dtype=float))
        pred_eval = np.log1p(work[pred_col].clip(lower=0).to_numpy(dtype=float))
        mae_name = "log_mae"
    elif transform == "raw":
        true_eval = work[true_col].to_numpy(dtype=float)
        pred_eval = work[pred_col].to_numpy(dtype=float)
        mae_name = "mae"
    else:
        raise ValueError(f"Unsupported transform: {transform}")

    slope, intercept = _slope_intercept(pred_eval, true_eval)
    row = {
        "source": source,
        "target": target,
        "n": int(len(work)),
        "spearman": float(work[true_col].corr(work[pred_col], method="spearman")),
        "pearson": float(pd.Series(true_eval).corr(pd.Series(pred_eval), method="pearson")),
        "r2": _r2(true_eval, pred_eval),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
    row[mae_name] = float(np.mean(np.abs(true_eval - pred_eval)))
    return row


def build_label_sanity(contract_path: Path) -> pd.DataFrame:
    df = pd.read_csv(contract_path)
    if "external_eval_ready" in df.columns:
        df = df[df["external_eval_ready"].astype(bool)].copy()

    rows = []
    rows.append(
        _metric_rows_for_pair(
            df,
            source="mal2025_overlap",
            target="popularity",
            true_col="anilist_popularity",
            pred_col="external_popularity_members",
            transform="log1p",
        )
    )
    rows.append(
        _metric_rows_for_pair(
            df,
            source="mal2025_overlap",
            target="meanScore",
            true_col="anilist_meanScore",
            pred_col="external_score_0_100",
            transform="raw",
        )
    )
    return pd.DataFrame(rows)


def _prediction_target_columns(target: str) -> tuple[str, str, str]:
    if target == "popularity":
        return "external_popularity_members", "prediction_popularity", "log1p"
    if target == "meanScore":
        return "external_score_0_100", "prediction_meanScore", "raw"
    raise ValueError(target)


def build_calibration_outputs(
    specs: Iterable[dict],
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    bin_rows = []
    for spec in specs:
        path = _resolve(spec["path"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for target in spec["targets"]:
            true_col, pred_col, transform = _prediction_target_columns(target)
            if true_col not in df.columns or pred_col not in df.columns:
                continue
            summary = _metric_rows_for_pair(
                df,
                source=f"{spec['exam']}:{spec['variant']}",
                target=target,
                true_col=true_col,
                pred_col=pred_col,
                transform=transform,
            )
            summary_rows.append(summary)

            work = df[[true_col, pred_col]].dropna().copy()
            work = work.sort_values(pred_col).reset_index(drop=True)
            bins = pd.qcut(work[pred_col].rank(method="first"), q=n_bins, labels=False)
            work["pred_quantile"] = bins.astype(int) + 1
            for q, group in work.groupby("pred_quantile", sort=True):
                true_raw = group[true_col].to_numpy(dtype=float)
                pred_raw = group[pred_col].to_numpy(dtype=float)
                row = {
                    "exam": spec["exam"],
                    "variant": spec["variant"],
                    "target": target,
                    "pred_quantile": int(q),
                    "n": int(len(group)),
                    "pred_mean": float(np.mean(pred_raw)),
                    "actual_mean": float(np.mean(true_raw)),
                    "pred_median": float(np.median(pred_raw)),
                    "actual_median": float(np.median(true_raw)),
                }
                if target == "popularity":
                    row["mean_log_error"] = float(np.mean(np.log1p(pred_raw) - np.log1p(true_raw)))
                    row["actual_to_pred_ratio"] = float(np.mean(true_raw) / max(float(np.mean(pred_raw)), 1e-12))
                else:
                    row["mean_error"] = float(np.mean(pred_raw - true_raw))
                    row["mae"] = float(np.mean(np.abs(pred_raw - true_raw)))
                bin_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(bin_rows)


def _write_markdown(
    label_sanity: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    path: Path,
) -> None:
    def table(df: pd.DataFrame) -> str:
        return df.replace({np.nan: ""}).to_markdown(index=False)

    lines = [
        "# MAL 2025 External Diagnostics",
        "",
        "## MAL 2025 overlap label sanity",
        "",
        "This check uses the MAL 2025 overlap rows, not the earlier MAL July label-check file.",
        "It verifies whether MAL `members` and `score * 10` are aligned with AniList labels before using MAL-only rows as an external exam.",
        "",
        table(label_sanity),
        "",
        "## Run22 external calibration summary",
        "",
        "Calibration is computed on existing Run22 prediction CSVs. Popularity is evaluated in `log1p` space; meanScore is evaluated in raw 0-100 score space.",
        "",
        table(calibration_summary),
        "",
        "## Prediction-quantile calibration bins",
        "",
        "Rows are grouped by predicted value quantiles. Monotonic actual means indicate useful ranking transfer; systematic gaps between predicted and actual means indicate scale mismatch.",
        "",
        table(calibration_bins),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        default="data/external_transformed/mal2025_image_external_eval_contract.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/experiments/sample_alignment",
    )
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_sanity = build_label_sanity(_resolve(args.contract))
    calibration_summary, calibration_bins = build_calibration_outputs(PREDICTION_SPECS, args.bins)

    label_sanity.to_csv(output_dir / "mal2025_overlap_label_sanity.csv", index=False, lineterminator="\n")
    calibration_summary.to_csv(output_dir / "mal2025_external_calibration_summary.csv", index=False, lineterminator="\n")
    calibration_bins.to_csv(output_dir / "mal2025_external_calibration_bins.csv", index=False, lineterminator="\n")
    _write_markdown(
        label_sanity,
        calibration_summary,
        calibration_bins,
        output_dir / "mal2025_external_diagnostics.md",
    )


if __name__ == "__main__":
    main()
