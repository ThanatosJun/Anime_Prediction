"""
Generate research-question-oriented EDA outputs.

Outputs:
- data/eda/rq_eda_summary.json
- data/eda/rq_eda_summary.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
EDA_DIR = Path("data/eda")

RAW_PICKLE = RAW_DIR / "anilist_anime_data_complete.pkl"
RAW_CSV = RAW_DIR / "anilist_anime_data_complete.csv"
PROCESSED_CSV = PROCESSED_DIR / "anilist_anime_data_processed_v1.csv"

RQ_SUMMARY_JSON = EDA_DIR / "rq_eda_summary.json"
RQ_SUMMARY_MD = EDA_DIR / "rq_eda_summary.md"
PERMUTATION_ROUNDS = 400
BOOTSTRAP_ROUNDS = 400
RNG_SEED = 42


def _load_raw() -> pd.DataFrame:
    if RAW_PICKLE.exists():
        return pd.read_pickle(RAW_PICKLE)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    raise FileNotFoundError("No supported raw dataset found in data/raw (expected PKL or CSV).")


def _load_processed() -> pd.DataFrame:
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError("Processed dataset not found. Run scripts/build_processed_dataset.py first.")
    return pd.read_csv(PROCESSED_CSV)


def _missing_ratio(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    return float(df[column].isna().mean())


def _corr_safe(a: pd.Series, b: pd.Series) -> float | None:
    corr = pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"))
    return None if pd.isna(corr) else float(corr)


def _safe_available_ratio(series: pd.Series) -> float:
    return float(series.notna().mean())


def _safe_non_empty_json_ratio(series: pd.Series) -> float:
    cleaned = series.astype(str).str.strip()
    non_empty = cleaned.notna() & (cleaned != "") & (cleaned != "[]") & (cleaned != "{}") & (cleaned != "nan")
    return float(non_empty.mean())


def _snapshot_control_metrics(processed_df: pd.DataFrame) -> dict:
    snapshot_control = {}
    if "release_year" in processed_df.columns:
        snapshot_control["corr_release_year_vs_popularity_raw"] = _corr_safe(
            processed_df["release_year"], processed_df["popularity"]
        )
    if "release_year" in processed_df.columns and "popularity_quarter_pct" in processed_df.columns:
        snapshot_control["corr_release_year_vs_popularity_quarter_pct"] = _corr_safe(
            processed_df["release_year"], processed_df["popularity_quarter_pct"]
        )

    raw_corr = snapshot_control.get("corr_release_year_vs_popularity_raw")
    normalized_corr = snapshot_control.get("corr_release_year_vs_popularity_quarter_pct")
    if raw_corr is not None and normalized_corr is not None:
        snapshot_control["absolute_corr_reduction"] = abs(raw_corr) - abs(normalized_corr)
    else:
        snapshot_control["absolute_corr_reduction"] = None
    return snapshot_control


def _bucket_balance_by_split(processed_df: pd.DataFrame) -> dict:
    if "split_pre_release_effective" not in processed_df.columns or "popularity_quarter_bucket" not in processed_df.columns:
        return {}
    subset = processed_df[processed_df["is_model_split"] == True].copy() if "is_model_split" in processed_df.columns else processed_df
    table = pd.crosstab(subset["split_pre_release_effective"], subset["popularity_quarter_bucket"], normalize="index")
    output = {}
    for split_name, row in table.iterrows():
        output[str(split_name)] = {str(col): float(val) for col, val in row.to_dict().items()}
    return output


def _multimodal_coverage_by_split(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    if "id" not in raw_df.columns or "id" not in processed_df.columns or "split_pre_release_effective" not in processed_df.columns:
        return {}
    raw_subset = raw_df[["id", "description", "coverImage_medium", "trailer_id"]].copy()
    merged = processed_df[["id", "split_pre_release_effective"]].merge(raw_subset, on="id", how="left")
    result = {}
    for split_name, group in merged.groupby("split_pre_release_effective", dropna=False):
        result[str(split_name)] = {
            "text_description_available_ratio": _safe_available_ratio(group["description"]),
            "image_cover_available_ratio": _safe_available_ratio(group["coverImage_medium"]),
            "trailer_id_available_ratio": _safe_available_ratio(group["trailer_id"]),
        }
    return result


def _multimodal_split_frame(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"id", "split_pre_release_effective"}
    if not required_cols.issubset(processed_df.columns) or "id" not in raw_df.columns:
        return pd.DataFrame()
    raw_subset = raw_df[["id", "description", "coverImage_medium", "bannerImage", "trailer_id"]].copy()
    merged = processed_df[["id", "split_pre_release_effective"]].merge(raw_subset, on="id", how="left")
    merged["text_available"] = merged["description"].notna().astype(int)
    merged["cover_available"] = merged["coverImage_medium"].notna().astype(int)
    merged["banner_available"] = merged["bannerImage"].notna().astype(int)
    merged["trailer_available"] = merged["trailer_id"].notna().astype(int)
    return merged


def _bucket_balance_permutation_test(processed_df: pd.DataFrame) -> dict:
    required_cols = {"split_pre_release_effective", "popularity_quarter_bucket"}
    if not required_cols.issubset(processed_df.columns):
        return {"available": False}

    subset = processed_df[
        processed_df["split_pre_release_effective"].isin(["train", "val", "test"])
        & processed_df["popularity_quarter_bucket"].notna()
    ][["split_pre_release_effective", "popularity_quarter_bucket"]].copy()
    if subset.empty:
        return {"available": False}

    rng = np.random.default_rng(RNG_SEED)

    def _compute_max_tvd(frame: pd.DataFrame) -> tuple[float, dict]:
        counts = pd.crosstab(frame["split_pre_release_effective"], frame["popularity_quarter_bucket"])
        row_ratios = counts.div(counts.sum(axis=1), axis=0)
        global_ratio = counts.sum(axis=0) / counts.values.sum()
        tvd = 0.5 * (row_ratios.sub(global_ratio, axis=1).abs().sum(axis=1))
        return float(tvd.max()), {str(k): float(v) for k, v in tvd.to_dict().items()}

    observed, per_split_tvd = _compute_max_tvd(subset)
    split_values = subset["split_pre_release_effective"].to_numpy(copy=True)

    extreme = 0
    for _ in range(PERMUTATION_ROUNDS):
        permuted = subset.copy()
        permuted["split_pre_release_effective"] = rng.permutation(split_values)
        stat, _ = _compute_max_tvd(permuted)
        if stat >= observed:
            extreme += 1
    p_value = (extreme + 1) / (PERMUTATION_ROUNDS + 1)
    return {
        "available": True,
        "method": "permutation_test_max_tvd",
        "rounds": PERMUTATION_ROUNDS,
        "observed_stat": observed,
        "p_value": p_value,
        "per_split_tvd": per_split_tvd,
    }


def _coverage_gap_permutation_test(multimodal_frame: pd.DataFrame, col_name: str) -> dict:
    if multimodal_frame.empty or col_name not in multimodal_frame.columns:
        return {"available": False}
    subset = multimodal_frame[multimodal_frame["split_pre_release_effective"].isin(["train", "val", "test"])].copy()
    if subset.empty:
        return {"available": False}

    means = subset.groupby("split_pre_release_effective")[col_name].mean()
    observed = float(means.max() - means.min())
    split_values = subset["split_pre_release_effective"].to_numpy(copy=True)
    rng = np.random.default_rng(RNG_SEED)

    extreme = 0
    for _ in range(PERMUTATION_ROUNDS):
        permuted = subset.copy()
        permuted["split_pre_release_effective"] = rng.permutation(split_values)
        perm_means = permuted.groupby("split_pre_release_effective")[col_name].mean()
        stat = float(perm_means.max() - perm_means.min())
        if stat >= observed:
            extreme += 1

    p_value = (extreme + 1) / (PERMUTATION_ROUNDS + 1)
    return {
        "available": True,
        "method": "permutation_test_max_mean_gap",
        "rounds": PERMUTATION_ROUNDS,
        "observed_stat": observed,
        "p_value": p_value,
        "split_means": {str(k): float(v) for k, v in means.to_dict().items()},
    }


def _snapshot_bootstrap_ci(processed_df: pd.DataFrame) -> dict:
    required_cols = {"release_year", "popularity", "popularity_quarter_pct"}
    if not required_cols.issubset(processed_df.columns):
        return {"available": False}
    subset = processed_df[list(required_cols)].dropna()
    if subset.empty:
        return {"available": False}

    rng = np.random.default_rng(RNG_SEED)
    reductions = []
    for _ in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, len(subset), len(subset))
        sample = subset.iloc[idx]
        raw_corr = _corr_safe(sample["release_year"], sample["popularity"])
        norm_corr = _corr_safe(sample["release_year"], sample["popularity_quarter_pct"])
        if raw_corr is None or norm_corr is None:
            continue
        reductions.append(abs(raw_corr) - abs(norm_corr))

    if not reductions:
        return {"available": False}
    lower, upper = np.percentile(reductions, [2.5, 97.5])
    return {
        "available": True,
        "method": "bootstrap_ci_abs_corr_reduction",
        "rounds": BOOTSTRAP_ROUNDS,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "mean_reduction": float(np.mean(reductions)),
    }


def _statistical_tests(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    multimodal_frame = _multimodal_split_frame(raw_df, processed_df)
    return {
        "bucket_balance_permutation": _bucket_balance_permutation_test(processed_df),
        "multimodal_coverage_permutation": {
            "text_available": _coverage_gap_permutation_test(multimodal_frame, "text_available"),
            "cover_available": _coverage_gap_permutation_test(multimodal_frame, "cover_available"),
            "banner_available": _coverage_gap_permutation_test(multimodal_frame, "banner_available"),
            "trailer_available": _coverage_gap_permutation_test(multimodal_frame, "trailer_available"),
        },
        "snapshot_reduction_bootstrap": _snapshot_bootstrap_ci(processed_df),
    }


def build_summary(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    coverage = {
        "description_missing_ratio": _missing_ratio(raw_df, "description"),
        "coverImage_medium_missing_ratio": _missing_ratio(raw_df, "coverImage_medium"),
        "trailer_id_missing_ratio": _missing_ratio(raw_df, "trailer_id"),
        "studios_missing_ratio": _missing_ratio(processed_df, "studios"),
        "genres_missing_ratio": _missing_ratio(processed_df, "genres"),
    }

    snapshot_control = _snapshot_control_metrics(processed_df)

    rq1_proxy = {
        "metadata_relation_coverage": {
            "studios_available_ratio": None
            if coverage["studios_missing_ratio"] is None
            else 1.0 - coverage["studios_missing_ratio"],
            "genres_available_ratio": None
            if coverage["genres_missing_ratio"] is None
            else 1.0 - coverage["genres_missing_ratio"],
            "studios_non_empty_ratio": _safe_non_empty_json_ratio(processed_df["studios"])
            if "studios" in processed_df.columns
            else None,
            "genres_non_empty_ratio": _safe_non_empty_json_ratio(processed_df["genres"])
            if "genres" in processed_df.columns
            else None,
        },
        "split_distribution": processed_df["split_pre_release_effective"].value_counts(dropna=False).to_dict()
        if "split_pre_release_effective" in processed_df.columns
        else {},
        "popularity_bucket_balance_by_split": _bucket_balance_by_split(processed_df),
    }

    rq2_proxy = {
        "multimodal_source_coverage": {
            "text_description_available_ratio": None
            if coverage["description_missing_ratio"] is None
            else 1.0 - coverage["description_missing_ratio"],
            "image_cover_available_ratio": None
            if coverage["coverImage_medium_missing_ratio"] is None
            else 1.0 - coverage["coverImage_medium_missing_ratio"],
            "trailer_id_available_ratio": None
            if coverage["trailer_id_missing_ratio"] is None
            else 1.0 - coverage["trailer_id_missing_ratio"],
        },
        "multimodal_coverage_by_split": _multimodal_coverage_by_split(raw_df, processed_df),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_raw": int(len(raw_df)),
        "rows_processed": int(len(processed_df)),
        "coverage": coverage,
        "snapshot_control": snapshot_control,
        "statistical_tests": _statistical_tests(raw_df, processed_df),
        "rq1_retrieval_proxy": rq1_proxy,
        "rq2_multimodal_proxy": rq2_proxy,
    }


def write_outputs(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    RQ_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RQ-oriented EDA Summary",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Raw rows: `{summary['rows_raw']}`",
        f"- Processed rows: `{summary['rows_processed']}`",
        "",
        "## Snapshot Control Evidence",
        "",
    ]
    sc = summary["snapshot_control"]
    lines.append(f"- Corr(release_year, popularity_raw): `{sc.get('corr_release_year_vs_popularity_raw')}`")
    lines.append(
        f"- Corr(release_year, popularity_quarter_pct): `{sc.get('corr_release_year_vs_popularity_quarter_pct')}`"
    )
    lines.append(f"- Absolute correlation reduction: `{sc.get('absolute_corr_reduction')}`")

    lines.extend(["", "## Statistical Test Layer", ""])
    tests = summary.get("statistical_tests", {})
    bucket_test = tests.get("bucket_balance_permutation", {})
    if bucket_test.get("available"):
        lines.append(
            "- Bucket balance permutation test "
            f"(max TVD): stat=`{bucket_test.get('observed_stat')}`, p=`{bucket_test.get('p_value')}`"
        )
    snapshot_ci = tests.get("snapshot_reduction_bootstrap", {})
    if snapshot_ci.get("available"):
        lines.append(
            "- Snapshot reduction bootstrap CI95: "
            f"[`{snapshot_ci.get('ci95_lower')}`, `{snapshot_ci.get('ci95_upper')}`], "
            f"mean=`{snapshot_ci.get('mean_reduction')}`"
        )
    modality_tests = tests.get("multimodal_coverage_permutation", {})
    for key, value in modality_tests.items():
        if value.get("available"):
            lines.append(
                f"- Coverage permutation `{key}`: stat=`{value.get('observed_stat')}`, p=`{value.get('p_value')}`"
            )

    lines.extend(["", "## RQ1 Proxy (Retrieval/Metadata Readiness)", ""])
    rq1 = summary["rq1_retrieval_proxy"]
    md_cov = rq1["metadata_relation_coverage"]
    lines.append(f"- Studios available ratio: `{md_cov.get('studios_available_ratio')}`")
    lines.append(f"- Genres available ratio: `{md_cov.get('genres_available_ratio')}`")
    lines.append(f"- Studios non-empty ratio: `{md_cov.get('studios_non_empty_ratio')}`")
    lines.append(f"- Genres non-empty ratio: `{md_cov.get('genres_non_empty_ratio')}`")
    for split_name, count in rq1.get("split_distribution", {}).items():
        lines.append(f"- Split `{split_name}` rows: `{count}`")
    lines.append("")
    lines.append("### Popularity Bucket Balance by Split")
    for split_name, dist in rq1.get("popularity_bucket_balance_by_split", {}).items():
        lines.append(f"- `{split_name}`: {dist}")

    lines.extend(["", "## RQ2 Proxy (Multimodal Readiness)", ""])
    rq2 = summary["rq2_multimodal_proxy"]["multimodal_source_coverage"]
    lines.append(f"- Text description available ratio: `{rq2.get('text_description_available_ratio')}`")
    lines.append(f"- Image cover available ratio: `{rq2.get('image_cover_available_ratio')}`")
    lines.append(f"- Trailer id available ratio: `{rq2.get('trailer_id_available_ratio')}`")
    lines.append("")
    lines.append("### Multimodal Coverage by Split")
    for split_name, stats in summary["rq2_multimodal_proxy"].get("multimodal_coverage_by_split", {}).items():
        lines.append(f"- `{split_name}`: {stats}")

    RQ_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_df = _load_raw()
    processed_df = _load_processed()
    summary = build_summary(raw_df, processed_df)
    write_outputs(summary)
    print(f"Wrote {RQ_SUMMARY_JSON}")
    print(f"Wrote {RQ_SUMMARY_MD}")


if __name__ == "__main__":
    main()
