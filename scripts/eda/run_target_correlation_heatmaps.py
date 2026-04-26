"""
Generate target correlation heatmaps for processed dataset columns.

Outputs:
- reports/figures/corr_popularity_heatmap.png
- reports/figures/corr_meanscore_heatmap.png
- reports/target_correlation_summary.json
- reports/target_correlation_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_CSV = Path("data/processed/anilist_anime_data_processed_v1.csv")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"

OUT_POP_HEATMAP = FIG_DIR / "corr_popularity_heatmap.png"
OUT_SCORE_HEATMAP = FIG_DIR / "corr_meanscore_heatmap.png"
OUT_SUMMARY_JSON = REPORTS_DIR / "target_correlation_summary.json"
OUT_SUMMARY_MD = REPORTS_DIR / "target_correlation_summary.md"

EXCLUDE_FROM_FEATURES = {
    "id",
    "title_romaji",
    "title_english",
    "title_native",
    "genres",
    "studios",
    "release_date",
    "release_quarter_key",
}


def _load_processed() -> pd.DataFrame:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError("Processed CSV not found. Run scripts/pipeline/build_processed_dataset.py first.")
    return pd.read_csv(PROCESSED_CSV)


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    candidate_cols = [c for c in work.columns if c not in EXCLUDE_FROM_FEATURES]
    work = work[candidate_cols]

    # Normalize booleans before one-hot expansion.
    for col in work.columns:
        if work[col].dtype == bool:
            work[col] = work[col].astype(int)

    numeric_cols = work.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in work.columns if c not in numeric_cols]

    low_card_cat = []
    for col in categorical_cols:
        nunique = int(work[col].nunique(dropna=True))
        # Keep one-hot space readable while still covering key categorical features.
        if nunique <= 20:
            low_card_cat.append(col)

    numeric_part = work[numeric_cols].copy()
    cat_part = pd.get_dummies(work[low_card_cat], prefix=low_card_cat, dummy_na=True, dtype=int)
    matrix = pd.concat([numeric_part, cat_part], axis=1)
    return matrix


def _rank_target_correlations(matrix: pd.DataFrame, target: str, top_k: int = 25) -> pd.Series:
    if target not in matrix.columns:
        return pd.Series(dtype="float64")
    corr = matrix.corr(numeric_only=True)[target].dropna()
    corr = corr.drop(labels=[target], errors="ignore")
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_k)


def _plot_heatmap(series: pd.Series, title: str, out_path: Path) -> None:
    if series.empty:
        return

    fig_h = max(6, 0.28 * len(series))
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    values = series.values.reshape(-1, 1)
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks([0], labels=["correlation"])
    ax.set_yticks(range(len(series)), labels=series.index)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_summary(pop_corr: pd.Series, score_corr: pd.Series, matrix: pd.DataFrame) -> None:
    summary = {
        "source_file": PROCESSED_CSV.as_posix(),
        "encoded_feature_count": int(matrix.shape[1]),
        "top_correlations": {
            "popularity": {k: float(v) for k, v in pop_corr.items()},
            "meanScore": {k: float(v) for k, v in score_corr.items()},
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Target Correlation Summary",
        "",
        f"- Source: `{summary['source_file']}`",
        f"- Encoded feature count: `{summary['encoded_feature_count']}`",
        "",
        "## Top Correlations with `popularity`",
        "",
    ]
    for key, value in summary["top_correlations"]["popularity"].items():
        lines.append(f"- `{key}`: `{value:.4f}`")
    lines.extend(["", "## Top Correlations with `meanScore`", ""])
    for key, value in summary["top_correlations"]["meanScore"].items():
        lines.append(f"- `{key}`: `{value:.4f}`")

    OUT_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = _load_processed()
    matrix = _build_feature_matrix(df)
    pop_corr = _rank_target_correlations(matrix, "popularity")
    score_corr = _rank_target_correlations(matrix, "meanScore")

    _plot_heatmap(pop_corr, "Top Correlations with Popularity", OUT_POP_HEATMAP)
    _plot_heatmap(score_corr, "Top Correlations with MeanScore", OUT_SCORE_HEATMAP)
    _write_summary(pop_corr, score_corr, matrix)

    print(f"Wrote {OUT_POP_HEATMAP}")
    print(f"Wrote {OUT_SCORE_HEATMAP}")
    print(f"Wrote {OUT_SUMMARY_JSON}")
    print(f"Wrote {OUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
