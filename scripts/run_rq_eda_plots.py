"""
Generate paper-ready plots from RQ-oriented EDA outputs.

Outputs:
- data/eda/figures/rq_snapshot_control.png
- data/eda/figures/rq_split_bucket_balance.png
- data/eda/figures/rq_multimodal_coverage_by_split.png
- data/eda/figures/rq_figure_notes.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EDA_DIR = Path("data/eda")
FIG_DIR = EDA_DIR / "figures"
RQ_SUMMARY_JSON = EDA_DIR / "rq_eda_summary.json"

PLOT_SNAPSHOT = FIG_DIR / "rq_snapshot_control.png"
PLOT_SPLIT_BALANCE = FIG_DIR / "rq_split_bucket_balance.png"
PLOT_MODALITY = FIG_DIR / "rq_multimodal_coverage_by_split.png"
FIG_NOTES = FIG_DIR / "rq_figure_notes.md"


def _load_summary() -> dict:
    if not RQ_SUMMARY_JSON.exists():
        raise FileNotFoundError("RQ summary not found. Run scripts/run_rq_eda.py first.")
    return json.loads(RQ_SUMMARY_JSON.read_text(encoding="utf-8"))


def _plot_snapshot_control(summary: dict) -> None:
    sc = summary.get("snapshot_control", {})
    labels = ["raw_popularity", "quarter_normalized"]
    values = [
        abs(sc.get("corr_release_year_vs_popularity_raw") or 0.0),
        abs(sc.get("corr_release_year_vs_popularity_quarter_pct") or 0.0),
    ]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values)
    plt.title("Snapshot Bias Proxy: |corr(release_year, target)|")
    plt.ylabel("Absolute correlation")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(PLOT_SNAPSHOT, dpi=160)
    plt.close()


def _plot_split_bucket_balance(summary: dict) -> None:
    balance = summary.get("rq1_retrieval_proxy", {}).get("popularity_bucket_balance_by_split", {})
    if not balance:
        return

    df = pd.DataFrame(balance).T.fillna(0.0)
    df = df[["cold_0_25", "warm_25_50", "hot_50_75", "top_75_100"]]

    plt.figure(figsize=(8, 5))
    bottoms = pd.Series([0.0] * len(df), index=df.index)
    for col in df.columns:
        plt.bar(df.index, df[col], bottom=bottoms, label=col)
        bottoms += df[col]
    plt.title("Popularity Quarter Bucket Distribution by Split")
    plt.ylabel("Ratio")
    plt.ylim(0, 1.0)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOT_SPLIT_BALANCE, dpi=160)
    plt.close()


def _plot_multimodal_coverage(summary: dict) -> None:
    coverage = summary.get("rq2_multimodal_proxy", {}).get("multimodal_coverage_by_split", {})
    if not coverage:
        return

    df = pd.DataFrame(coverage).T.fillna(0.0)
    cols = [
        "text_description_available_ratio",
        "image_cover_available_ratio",
        "trailer_id_available_ratio",
    ]
    df = df[cols]

    ax = df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Multimodal Source Coverage by Split")
    ax.set_ylabel("Available ratio")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_MODALITY, dpi=160)
    plt.close()


def _write_figure_notes(summary: dict) -> None:
    sc = summary.get("snapshot_control", {})
    tests = summary.get("statistical_tests", {})
    bucket_test = tests.get("bucket_balance_permutation", {})
    snapshot_ci = tests.get("snapshot_reduction_bootstrap", {})
    modality_tests = tests.get("multimodal_coverage_permutation", {})
    lines = [
        "# RQ Figure Notes",
        "",
        "## Figure 1: Snapshot Control",
        f"- File: `{PLOT_SNAPSHOT.as_posix()}`",
        "- Meaning: compare absolute correlation between release year and target before/after quarter normalization.",
        f"- Observed reduction: `{sc.get('absolute_corr_reduction')}`",
    ]
    if snapshot_ci.get("available"):
        lines.append(
            "- Statistical support: bootstrap CI95 for reduction = "
            f"[`{snapshot_ci.get('ci95_lower')}`, `{snapshot_ci.get('ci95_upper')}`], "
            f"mean=`{snapshot_ci.get('mean_reduction')}`."
        )
    lines.extend([
        "",
        "## Figure 2: Split Bucket Balance",
        f"- File: `{PLOT_SPLIT_BALANCE.as_posix()}`",
        "- Meaning: verify whether popularity bucket classes remain balanced across train/val/test.",
    ])
    if bucket_test.get("available"):
        lines.append(
            "- Statistical support: permutation test (max TVD) "
            f"stat=`{bucket_test.get('observed_stat')}`, p=`{bucket_test.get('p_value')}`."
        )
    lines.extend([
        "",
        "## Figure 3: Multimodal Coverage by Split",
        f"- File: `{PLOT_MODALITY.as_posix()}`",
        "- Meaning: show text/image/trailer availability mismatch between splits for RQ2 risk discussion.",
        "",
        "### Statistical Support for Coverage Gaps",
    ])
    for key in ["text_available", "cover_available", "banner_available", "trailer_available"]:
        metric = modality_tests.get(key, {})
        if metric.get("available"):
            lines.append(
                f"- `{key}` permutation: stat=`{metric.get('observed_stat')}`, "
                f"p=`{metric.get('p_value')}`."
            )
    lines.extend([
        "",
    ])
    FIG_NOTES.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()

    _plot_snapshot_control(summary)
    _plot_split_bucket_balance(summary)
    _plot_multimodal_coverage(summary)
    _write_figure_notes(summary)

    print(f"Wrote {PLOT_SNAPSHOT}")
    print(f"Wrote {PLOT_SPLIT_BALANCE}")
    print(f"Wrote {PLOT_MODALITY}")
    print(f"Wrote {FIG_NOTES}")


if __name__ == "__main__":
    main()
