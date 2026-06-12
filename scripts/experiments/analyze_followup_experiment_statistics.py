"""Build follow-up statistics for experiment-review questions.

This script is read-only with respect to model artifacts. It uses existing
per-row prediction CSVs to compute paired bootstrap tests, ablation effect
sizes, and image-backbone/proxy diagnostics for the final-project follow-ups.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "experiments" / "sample_alignment"


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    path: Path
    pred_col: str = "pred"
    target_col: str = "target"


@dataclass(frozen=True)
class PairedComparison:
    group: str
    target: str
    model_a: PredictionSpec
    model_b: PredictionSpec
    metrics: tuple[str, ...]
    note: str


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_predictions(spec: PredictionSpec) -> pd.DataFrame:
    path = _resolve(spec.path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "id" not in df.columns or spec.pred_col not in df.columns or spec.target_col not in df.columns:
        raise ValueError(f"{path} does not contain id/{spec.pred_col}/{spec.target_col}")
    out = df[["id", spec.target_col, spec.pred_col]].copy()
    out.columns = ["id", "target", spec.name]
    out["id"] = out["id"].astype(str)
    return out.dropna()


def _aligned_pair(a: PredictionSpec, b: PredictionSpec) -> pd.DataFrame:
    left = _load_predictions(a)
    right = _load_predictions(b)[["id", "target", b.name]]
    merged = left.merge(right, on="id", how="inner", suffixes=("_a", "_b"))
    merged["target"] = merged["target_a"]
    return merged[["id", "target", a.name, b.name]].dropna()


def _eval_arrays(df: pd.DataFrame, pred_col: str, target: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    y = df["target"].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    if target == "popularity" and metric in {"log_mae", "log_r2", "factor_acc_2x"}:
        y = np.log1p(np.clip(y, 0, None))
        pred = np.log1p(np.clip(pred, 0, None))
    return y, pred


def _metric_value(df: pd.DataFrame, pred_col: str, target: str, metric: str) -> float:
    y, pred = _eval_arrays(df, pred_col, target, metric)
    if metric in {"log_mae", "mae"}:
        return float(np.mean(np.abs(pred - y)))
    if metric in {"log_r2", "r2"}:
        denom = float(np.sum((y - y.mean()) ** 2))
        return float("nan") if denom == 0 else float(1 - np.sum((y - pred) ** 2) / denom)
    if metric == "factor_acc_2x":
        return float(np.mean(np.abs(pred - y) < np.log(2)))
    if metric == "acc_within_10pt":
        return float(np.mean(np.abs(pred - y) < 10))
    if metric == "spearman":
        return float(pd.Series(y).corr(pd.Series(pred), method="spearman"))
    raise ValueError(metric)


def _metric_direction(metric: str) -> str:
    if metric in {"log_mae", "mae"}:
        return "lower"
    return "higher"


def _better_delta(df: pd.DataFrame, a_col: str, b_col: str, target: str, metric: str) -> float:
    a_val = _metric_value(df, a_col, target, metric)
    b_val = _metric_value(df, b_col, target, metric)
    if _metric_direction(metric) == "lower":
        return b_val - a_val
    return a_val - b_val


def _bootstrap_comparison(
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    target: str,
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
) -> dict:
    n = len(df)
    a_val = _metric_value(df, a_col, target, metric)
    b_val = _metric_value(df, b_col, target, metric)
    observed = _better_delta(df, a_col, b_col, target, metric)
    deltas = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample = df.iloc[sample_idx]
        deltas[idx] = _better_delta(sample, a_col, b_col, target, metric)

    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    p_left = float(np.mean(deltas <= 0))
    p_right = float(np.mean(deltas >= 0))
    p_two_sided = min(1.0, 2 * min(p_left, p_right))
    return {
        "n": n,
        "metric": metric,
        "direction": _metric_direction(metric),
        "model_a_value": a_val,
        "model_b_value": b_val,
        "delta_a_better": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_two_sided": p_two_sided,
        "a_better_bootstrap_rate": float(np.mean(deltas > 0)),
    }


def _spec_for_run(run_id: str, target: str, name: str | None = None) -> PredictionSpec:
    return PredictionSpec(
        name=name or run_id,
        path=Path("final_project") / "runs" / run_id / target / "pred_test.csv",
    )


def _spec_for_baseline(model: str, target: str) -> PredictionSpec:
    return PredictionSpec(
        name=model,
        path=Path("reports")
        / "experiments"
        / "sample_alignment"
        / "carma_tensor_predictions"
        / "test"
        / model
        / target
        / "predictions.csv",
        pred_col="prediction",
    )


def _spec_for_external_carma(split: str, target: str) -> PredictionSpec:
    pred_col = f"prediction_{target}"
    target_col = "external_popularity_members" if target == "popularity" else "external_score_0_100"
    return PredictionSpec(
        name="CARMA-Run22",
        path=Path("data") / "external_transformed" / f"run22_{split}_predictions.csv",
        pred_col=pred_col,
        target_col=target_col,
    )


def _spec_for_external_baseline(split: str, model: str, target: str) -> PredictionSpec:
    return PredictionSpec(
        name=model,
        path=Path("reports")
        / "experiments"
        / "sample_alignment"
        / "carma_tensor_predictions"
        / split
        / model
        / target
        / "predictions.csv",
        pred_col="prediction",
    )


def build_paired_tests(n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comparisons: list[PairedComparison] = []

    exp1_models = [
        "F1-RF-Meta-CARMATensor",
        "F2-XGB-Concat-CARMATensor",
        "C1-Armenta-CARMATensor",
        "C2-CrossAttention-CARMATensor",
        "C2-RecurrentFusion-CARMATensor",
        "C3-RAG-XGB-CARMATensor",
    ]
    for target in ("popularity", "meanScore"):
        metrics = ("log_mae", "spearman") if target == "popularity" else ("mae", "spearman")
        for model in exp1_models:
            comparisons.append(
                PairedComparison(
                    group="exp1_full_temporal_test",
                    target=target,
                    model_a=_spec_for_run("22", target, name="CARMA-Run22"),
                    model_b=_spec_for_baseline(model, target),
                    metrics=metrics,
                    note="CARMA Run22 and the baseline are evaluated on the same available full temporal test ids.",
                )
            )

    ablations = [
        ("abl_rag_off_pt", "remove retrieval"),
        ("abl_no_image_pt", "remove image"),
        ("abl_full_notrend_pt", "remove temporal trend"),
        ("abl_img_cover_pt", "cover-only image branch"),
        ("abl_img_banner_pt", "banner-only image branch"),
        ("abl_img_yolo_pt", "YOLO-only image branch"),
        ("abl_img_cover_banner_pt", "cover+banner image branch"),
    ]
    for target in ("popularity", "meanScore"):
        metrics = ("log_mae", "spearman") if target == "popularity" else ("mae", "spearman")
        for run_id, note in ablations:
            comparisons.append(
                PairedComparison(
                    group="exp2_ablation",
                    target=target,
                    model_a=_spec_for_run("abl_full_pt", target, name="CARMA-full-ablation"),
                    model_b=_spec_for_run(run_id, target, name=run_id),
                    metrics=metrics,
                    note=note,
                )
            )

    rows = []
    for comparison in comparisons:
        try:
            df = _aligned_pair(comparison.model_a, comparison.model_b)
        except FileNotFoundError:
            continue
        for metric in comparison.metrics:
            stats = _bootstrap_comparison(
                df,
                comparison.model_a.name,
                comparison.model_b.name,
                comparison.target,
                metric,
                rng,
                n_boot,
            )
            rows.append(
                {
                    "group": comparison.group,
                    "target": comparison.target,
                    "model_a": comparison.model_a.name,
                    "model_b": comparison.model_b.name,
                    "note": comparison.note,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def build_external_paired_tests(n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    baseline_models = [
        "F2-XGB-Concat-CARMATensor",
        "C2-CrossAttention-CARMATensor",
        "C2-RecurrentFusion-CARMATensor",
        "C3-RAG-XGB-CARMATensor",
    ]
    external_splits = {
        "mal2025_popularity_local_ready": {
            "targets": ("popularity",),
            "variant": "no_yolo",
            "priority": "main",
        },
        "mal2025_dual_local_ready": {
            "targets": ("popularity", "meanScore"),
            "variant": "no_yolo",
            "priority": "main",
        },
        "mal2025_popularity_local_ready_yolo": {
            "targets": ("popularity",),
            "variant": "cover_yolo",
            "priority": "main",
        },
        "mal2025_dual_local_ready_yolo": {
            "targets": ("popularity", "meanScore"),
            "variant": "cover_yolo",
            "priority": "main",
        },
        "mal2025_popularity_local_ready_yolo_coverbanner": {
            "targets": ("popularity",),
            "variant": "cover_yolo_coverbanner_proxy",
            "priority": "diagnostic",
        },
        "mal2025_dual_local_ready_yolo_coverbanner": {
            "targets": ("popularity", "meanScore"),
            "variant": "cover_yolo_coverbanner_proxy",
            "priority": "diagnostic",
        },
    }

    rows = []
    for split, split_info in external_splits.items():
        for target in split_info["targets"]:
            metrics = ("log_mae", "spearman") if target == "popularity" else ("mae", "spearman")
            for model in baseline_models:
                carma = _spec_for_external_carma(split, target)
                baseline = _spec_for_external_baseline(split, model, target)
                try:
                    df = _aligned_pair(carma, baseline)
                except FileNotFoundError:
                    continue
                for metric in metrics:
                    stats = _bootstrap_comparison(
                        df,
                        carma.name,
                        baseline.name,
                        target,
                        metric,
                        rng,
                        n_boot,
                    )
                    rows.append(
                        {
                            "group": "exp3_external_paired",
                            "split": split,
                            "variant": split_info["variant"],
                            "priority": split_info["priority"],
                            "target": target,
                            "model_a": carma.name,
                            "model_b": baseline.name,
                            "note": "CARMA Run22 and the baseline are evaluated on the same MAL 2025 external rows.",
                            **stats,
                        }
                    )
    return pd.DataFrame(rows)


def build_image_proxy_diagnostics(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exp2 = paired[paired["group"].eq("exp2_ablation")].copy()
    for _, row in exp2.iterrows():
        if row["model_b"] not in {
            "abl_img_cover_pt",
            "abl_img_banner_pt",
            "abl_img_yolo_pt",
            "abl_img_cover_banner_pt",
            "abl_no_image_pt",
        }:
            continue
        rows.append(
            {
                "source": "carma_image_source_ablation",
                "target": row["target"],
                "comparison": f"{row['model_a']} vs {row['model_b']}",
                "metric": row["metric"],
                "n": int(row["n"]),
                "full_value": row["model_a_value"],
                "proxy_or_ablation_value": row["model_b_value"],
                "delta_full_better": row["delta_a_better"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "p_two_sided": row["p_two_sided"],
                "claim_boundary": "Image-source ablation inside CARMA; not a CNN-vs-Swin replacement.",
            }
        )

    baseline_path = ROOT / "reports" / "baselines" / "reference_baseline_results.csv"
    if baseline_path.exists():
        baselines = pd.read_csv(baseline_path)
        if {"model", "target", "n_test"}.issubset(baselines.columns):
            keep = baselines[
                baselines["model"].astype(str).str.contains("ResNet50|CTNNDualVisual", regex=True, na=False)
            ].copy()
            metric_cols = {
                "test_MAE": "MAE",
                "test_Spearman_rho": "spearman_rho",
                "test_log_MAE": "log_MAE",
            }
            for _, row in keep.iterrows():
                for col, metric in metric_cols.items():
                    if col not in row or pd.isna(row[col]):
                        continue
                    rows.append(
                        {
                            "source": "literature_proxy_reference",
                            "target": row["target"],
                            "comparison": row["model"],
                            "metric": metric,
                            "n": int(row["n_test"]),
                            "full_value": np.nan,
                            "proxy_or_ablation_value": row[col],
                            "delta_full_better": np.nan,
                            "ci95_low": np.nan,
                            "ci95_high": np.nan,
                            "p_two_sided": np.nan,
                            "claim_boundary": "ResNet/CNN reference baseline only; not paired with CARMA internals.",
                        }
                    )
    return pd.DataFrame(rows)


def _write_markdown(
    paired: pd.DataFrame,
    external_paired: pd.DataFrame,
    image_diag: pd.DataFrame,
    path: Path,
) -> None:
    def table(df: pd.DataFrame) -> str:
        markdown = df.replace({np.nan: ""}).to_markdown(index=False)
        return "\n".join(line.rstrip() for line in markdown.splitlines())

    lines = [
        "# Follow-up Experiment Statistics",
        "",
        "This report uses existing per-row prediction artifacts. Positive `delta_a_better` means `model_a` is better than `model_b` under the listed metric direction.",
        "",
        "## Internal paired bootstrap tests",
        "",
        table(paired),
        "",
        "## External paired bootstrap tests",
        "",
        "External tests compare CARMA Run22 with tensor-aligned baselines on the same MAL 2025 rows. `cover_yolo_coverbanner_proxy` rows are diagnostic only.",
        "",
        table(external_paired),
        "",
        "## Image backbone/proxy diagnostics",
        "",
        "The CARMA artifacts support image-source ablations and literature-proxy comparisons. They do not support a strict one-variable CNN-vs-Swin replacement inside the same CARMA architecture.",
        "",
        table(image_diag),
        "",
    ]
    text = "\n".join(lines)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    _write_text_lf(path, text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paired = build_paired_tests(n_boot=args.n_boot, seed=args.seed)
    external_paired = build_external_paired_tests(n_boot=args.n_boot, seed=args.seed)
    image_diag = build_image_proxy_diagnostics(paired)

    paired_path = args.out_dir / "followup_paired_bootstrap_tests.csv"
    external_paired_path = args.out_dir / "followup_external_paired_bootstrap_tests.csv"
    external_paired_md_path = args.out_dir / "followup_external_paired_bootstrap_tests.md"
    image_path = args.out_dir / "followup_image_proxy_diagnostics.csv"
    report_path = args.out_dir / "followup_experiment_statistics.md"
    paired.to_csv(paired_path, index=False)
    external_paired.to_csv(external_paired_path, index=False)
    external_paired_md = external_paired.replace({np.nan: ""}).to_markdown(index=False)
    external_paired_md = "\n".join(line.rstrip() for line in external_paired_md.splitlines())
    _write_text_lf(external_paired_md_path, external_paired_md + "\n")
    image_diag.to_csv(image_path, index=False)
    _write_markdown(paired, external_paired, image_diag, report_path)

    print(f"Wrote {paired_path}")
    print(f"Wrote {external_paired_path}")
    print(f"Wrote {external_paired_md_path}")
    print(f"Wrote {image_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
