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

LOCAL_READY_BY_EXAM = {
    "pop_only": "data/external_transformed/mal2025_image_mal_only_popularity_exam_local_ready.csv",
    "dual": "data/external_transformed/mal2025_image_mal_only_dual_target_exam_local_ready.csv",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


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


def _safe_spearman(true_values: pd.Series, pred_values: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "true": pd.to_numeric(true_values, errors="coerce"),
            "pred": pd.to_numeric(pred_values, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame["true"].corr(frame["pred"], method="spearman"))


def _load_prediction_with_metadata(spec: dict) -> pd.DataFrame:
    prediction = pd.read_csv(_resolve(spec["path"]))
    metadata_path = _resolve(LOCAL_READY_BY_EXAM[spec["exam"]])
    metadata_cols = ["external_exam_id", "format", "source", "release_year"]
    metadata = pd.read_csv(metadata_path, usecols=metadata_cols)
    return prediction.merge(metadata, on="external_exam_id", how="left")


def _quantile_labels(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(method="first")
    labels = [f"{prefix}{idx}_low" if idx == 1 else f"{prefix}{idx}" for idx in range(1, n_bins + 1)]
    labels[-1] = f"{prefix}{n_bins}_high"
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = ranked.notna()
    if valid.sum() == 0:
        return out
    out.loc[valid] = pd.qcut(ranked.loc[valid], q=n_bins, labels=labels, duplicates="drop").astype(str)
    return out


def _release_period(series: pd.Series) -> pd.Series:
    years = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        years,
        bins=[0, 2009, 2014, 2019, 2026],
        labels=["<=2009", "2010-2014", "2015-2019", "2020-2026"],
        include_lowest=True,
    ).astype("object")


def _top_or_other(series: pd.Series, top_n: int = 6) -> pd.Series:
    normalized = series.fillna("UNKNOWN").astype(str).str.strip()
    top_values = set(normalized.value_counts().head(top_n).index)
    return normalized.where(normalized.isin(top_values), "OTHER_SMALL")


def _metric_summary_for_slice(
    group: pd.DataFrame,
    target: str,
    true_col: str,
    pred_col: str,
) -> dict:
    work = group[[true_col, pred_col]].dropna()
    true_raw = work[true_col].to_numpy(dtype=float)
    pred_raw = work[pred_col].to_numpy(dtype=float)
    row = {
        "n": int(len(work)),
        "spearman": _safe_spearman(work[true_col], work[pred_col]),
        "actual_mean": float(np.mean(true_raw)) if len(work) else float("nan"),
        "pred_mean": float(np.mean(pred_raw)) if len(work) else float("nan"),
        "actual_median": float(np.median(true_raw)) if len(work) else float("nan"),
        "pred_median": float(np.median(pred_raw)) if len(work) else float("nan"),
    }
    if target == "popularity":
        true_eval = np.log1p(np.clip(true_raw, 0, None))
        pred_eval = np.log1p(np.clip(pred_raw, 0, None))
        row["metric"] = "log_mae"
        row["error"] = float(np.mean(np.abs(pred_eval - true_eval))) if len(work) else float("nan")
        row["signed_error"] = float(np.mean(pred_eval - true_eval)) if len(work) else float("nan")
        row["actual_to_pred_ratio"] = (
            float(np.mean(true_raw) / max(float(np.mean(pred_raw)), 1e-12)) if len(work) else float("nan")
        )
    else:
        row["metric"] = "mae"
        row["error"] = float(np.mean(np.abs(pred_raw - true_raw))) if len(work) else float("nan")
        row["signed_error"] = float(np.mean(pred_raw - true_raw)) if len(work) else float("nan")
        row["actual_to_pred_ratio"] = float("nan")
    return row


def build_error_slice_outputs(specs: Iterable[dict], n_bins: int) -> pd.DataFrame:
    rows = []
    for spec in specs:
        path = _resolve(spec["path"])
        if not path.exists():
            continue
        df = _load_prediction_with_metadata(spec)
        df["popularity_quantile"] = _quantile_labels(df["external_popularity_members"], n_bins, "Q")
        if "external_score_0_100" in df.columns:
            df["score_quantile"] = _quantile_labels(df["external_score_0_100"], n_bins, "Q")
        df["release_period"] = _release_period(df["release_year"])
        df["format_slice"] = _top_or_other(df["format"])
        df["source_slice"] = _top_or_other(df["source"])
        pop_threshold = pd.to_numeric(df["external_popularity_members"], errors="coerce").quantile(0.9)
        df["popularity_tail"] = np.where(
            pd.to_numeric(df["external_popularity_members"], errors="coerce") >= pop_threshold,
            "top_10pct",
            "lower_90pct",
        )

        slice_columns = [
            "popularity_quantile",
            "release_period",
            "format_slice",
            "source_slice",
            "popularity_tail",
        ]

        for target in spec["targets"]:
            true_col, pred_col, _ = _prediction_target_columns(target)
            if true_col not in df.columns or pred_col not in df.columns:
                continue
            target_slice_columns = list(slice_columns)
            if target == "meanScore" and "score_quantile" in df.columns:
                target_slice_columns.insert(1, "score_quantile")
            for slice_col in target_slice_columns:
                for slice_value, group in df.dropna(subset=[slice_col]).groupby(slice_col, sort=True):
                    if len(group) < 10:
                        continue
                    summary = _metric_summary_for_slice(group, target, true_col, pred_col)
                    rows.append(
                        {
                            "exam": spec["exam"],
                            "variant": spec["variant"],
                            "target": target,
                            "slice_type": slice_col,
                            "slice_value": str(slice_value),
                            **summary,
                        }
                    )
    return pd.DataFrame(rows)


def _case_row(
    row: pd.Series,
    variant: str,
    case_type: str,
    note: str,
) -> dict:
    out = {
        "variant": variant,
        "case_type": case_type,
        "mal_id": row.get("mal_id"),
        "title_romaji": row.get("title_romaji"),
        "format": row.get("format"),
        "source": row.get("source"),
        "release_year": row.get("release_year"),
        "note": note,
    }
    for col in [
        "external_popularity_members",
        "prediction_popularity",
        "pop_signed_log_error",
        "actual_to_pred_ratio",
        "external_score_0_100",
        "prediction_meanScore",
        "score_signed_error",
        "score_abs_error",
    ]:
        out[col] = row.get(col, np.nan)
    return out


def build_case_examples(specs: Iterable[dict], top_k: int = 5) -> pd.DataFrame:
    rows = []
    for spec in specs:
        if spec["exam"] != "dual" or spec["variant"] not in {"no_yolo", "cover_yolo"}:
            continue
        path = _resolve(spec["path"])
        if not path.exists():
            continue
        df = _load_prediction_with_metadata(spec)
        if "prediction_popularity" in df.columns:
            work = df.dropna(subset=["external_popularity_members", "prediction_popularity"]).copy()
            work["pop_signed_log_error"] = np.log1p(work["prediction_popularity"].clip(lower=0)) - np.log1p(
                work["external_popularity_members"].clip(lower=0)
            )
            work["actual_to_pred_ratio"] = work["external_popularity_members"] / work["prediction_popularity"].clip(lower=1e-12)
            actual_top = work["external_popularity_members"].quantile(0.9)
            pred_top = work["prediction_popularity"].quantile(0.9)

            success = work[
                (work["external_popularity_members"] >= actual_top)
                & (work["prediction_popularity"] >= pred_top)
            ].sort_values("external_popularity_members", ascending=False)
            for _, row in success.head(top_k).iterrows():
                rows.append(
                    _case_row(
                        row,
                        spec["variant"],
                        "popularity_ranking_success",
                        "Actual and predicted popularity are both in the top decile.",
                    )
                )

            under = work[work["external_popularity_members"] >= actual_top].sort_values("pop_signed_log_error")
            for _, row in under.head(top_k).iterrows():
                rows.append(
                    _case_row(
                        row,
                        spec["variant"],
                        "popularity_tail_underestimate",
                        "High-MAL-member title strongly underpredicted on the MAL scale.",
                    )
                )

        if "prediction_meanScore" in df.columns:
            work = df.dropna(subset=["external_score_0_100", "prediction_meanScore"]).copy()
            work["score_signed_error"] = work["prediction_meanScore"] - work["external_score_0_100"]
            work["score_abs_error"] = work["score_signed_error"].abs()
            score_top = work["external_score_0_100"].quantile(0.9)

            high_score_under = work[work["external_score_0_100"] >= score_top].sort_values("score_signed_error")
            for _, row in high_score_under.head(top_k).iterrows():
                rows.append(
                    _case_row(
                        row,
                        spec["variant"],
                        "score_high_underestimate",
                        "High-MAL-score title underpredicted by the model.",
                    )
                )

            large_error = work.sort_values("score_abs_error", ascending=False)
            for _, row in large_error.head(top_k).iterrows():
                rows.append(
                    _case_row(
                        row,
                        spec["variant"],
                        "score_largest_absolute_error",
                        "Largest absolute score error in the dual external split.",
                    )
                )

    return pd.DataFrame(rows)


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
    error_slices: pd.DataFrame,
    case_examples: pd.DataFrame,
    path: Path,
) -> None:
    def table(df: pd.DataFrame) -> str:
        markdown = df.replace({np.nan: ""}).to_markdown(index=False)
        return "\n".join(line.rstrip() for line in markdown.splitlines())

    slice_view = error_slices[
        error_slices["exam"].eq("dual")
        & error_slices["variant"].isin(["no_yolo", "cover_yolo"])
        & (
            (error_slices["target"].eq("popularity") & error_slices["slice_type"].isin(["popularity_quantile", "popularity_tail"]))
            | (error_slices["target"].eq("meanScore") & error_slices["slice_type"].isin(["score_quantile", "popularity_tail"]))
        )
    ].copy()
    case_view_columns = [
        "variant",
        "case_type",
        "title_romaji",
        "format",
        "source",
        "external_popularity_members",
        "prediction_popularity",
        "actual_to_pred_ratio",
        "external_score_0_100",
        "prediction_meanScore",
        "score_signed_error",
    ]
    case_view = case_examples[case_view_columns].copy()

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
        "## External error slices",
        "",
        "Slices are computed from existing Run22 prediction CSVs joined with MAL 2025 local-ready metadata. For popularity, `signed_error` is `log1p(prediction) - log1p(actual)`, so negative values indicate underestimation on the MAL member scale. For meanScore, `signed_error` is raw predicted score minus MAL score * 10.",
        "The compact table below shows the dual-target main variants; the full slice table, including format/source/release-year slices, is stored in `mal2025_external_error_slices.csv`.",
        "",
        table(slice_view),
        "",
        "## External case examples",
        "",
        "These examples are selected from dual-target no-YOLO and cover-derived YOLO variants. They are diagnostic examples, not additional evaluation rows. The full case table is stored in `mal2025_external_case_examples.csv`.",
        "",
        table(case_view),
        "",
    ]
    text = "\n".join(lines)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    _write_text_lf(path, text + "\n")


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
    error_slices = build_error_slice_outputs(PREDICTION_SPECS, args.bins)
    case_examples = build_case_examples(PREDICTION_SPECS)

    label_sanity.to_csv(output_dir / "mal2025_overlap_label_sanity.csv", index=False, lineterminator="\n")
    calibration_summary.to_csv(output_dir / "mal2025_external_calibration_summary.csv", index=False, lineterminator="\n")
    calibration_bins.to_csv(output_dir / "mal2025_external_calibration_bins.csv", index=False, lineterminator="\n")
    error_slices.to_csv(output_dir / "mal2025_external_error_slices.csv", index=False, lineterminator="\n")
    case_examples.to_csv(output_dir / "mal2025_external_case_examples.csv", index=False, lineterminator="\n")
    _write_markdown(
        label_sanity,
        calibration_summary,
        calibration_bins,
        error_slices,
        case_examples,
        output_dir / "mal2025_external_diagnostics.md",
    )


if __name__ == "__main__":
    main()
