"""
Build processed dataset with explicit outlier handling rules.

Input:
- latest `data/interim/anilist_anime_data_interim_*.csv`

Outputs:
- `data/processed/anilist_anime_data_processed_v1.csv`
- `data/processed/anilist_anime_data_processed_v1_meta.json`
- `data/eda/outlier_handling_summary.md`
- `data/eda/outlier_handling_summary.json`
- `data/eda/target_engineering_summary.md`
- `data/eda/target_engineering_summary.json`
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
TARGET_SUMMARY_JSON = EDA_DIR / "target_engineering_summary.json"
TARGET_SUMMARY_MD = EDA_DIR / "target_engineering_summary.md"
RULE_VERSION = "decision_eda_v3"
UNKNOWN_SPLIT_POLICY = "exclude_unknown_from_model_splits"

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
SEASON_TO_QUARTER = {"WINTER": 1, "SPRING": 2, "SUMMER": 3, "FALL": 4}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


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


def _derive_release_quarter(df: pd.DataFrame) -> None:
    season_series = (
        df["season"].astype(str).str.upper().str.strip()
        if "season" in df.columns
        else pd.Series(index=df.index, dtype="object")
    )
    season_quarter = season_series.map(SEASON_TO_QUARTER)

    month_quarter = pd.Series(index=df.index, dtype="float64")
    if "startDate_month" in df.columns:
        month_values = pd.to_numeric(df["startDate_month"], errors="coerce")
        month_quarter = ((month_values - 1) // 3 + 1).where(month_values.between(1, 12))

    release_quarter = season_quarter.fillna(month_quarter).astype("Int64")

    release_year = pd.Series(index=df.index, dtype="Int64")
    if "seasonYear" in df.columns:
        release_year = pd.to_numeric(df["seasonYear"], errors="coerce").astype("Int64")
    if "startDate_year" in df.columns:
        start_year = pd.to_numeric(df["startDate_year"], errors="coerce").astype("Int64")
        release_year = release_year.fillna(start_year)

    df["release_year"] = release_year
    df["release_quarter"] = release_quarter
    valid = release_year.notna() & release_quarter.notna()
    df["release_quarter_key"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[valid, "release_quarter_key"] = (
        df.loc[valid, "release_year"].astype(int).astype(str)
        + "Q"
        + df.loc[valid, "release_quarter"].astype(int).astype(str)
    )


def _add_popularity_quarter_target(df: pd.DataFrame) -> dict:
    if "popularity" not in df.columns:
        df["popularity_quarter_pct"] = pd.NA
        df["popularity_quarter_bucket"] = pd.NA
        return {"available": False}

    popularity = pd.to_numeric(df["popularity"], errors="coerce")
    quarter_key = df["release_quarter_key"]
    pct = popularity.groupby(quarter_key, dropna=False).rank(pct=True, ascending=True)
    df["popularity_quarter_pct"] = pct

    bins = [-0.000001, 0.25, 0.50, 0.75, 1.0]
    labels = ["cold_0_25", "warm_25_50", "hot_50_75", "top_75_100"]
    df["popularity_quarter_bucket"] = pd.cut(pct, bins=bins, labels=labels)
    df.loc[pct.isna(), "popularity_quarter_bucket"] = pd.NA

    bucket_counts = (
        df["popularity_quarter_bucket"]
        .value_counts(dropna=False)
        .rename_axis("bucket")
        .reset_index(name="count")
    )
    group_counts = (
        quarter_key.value_counts(dropna=False)
        .rename_axis("release_quarter_key")
        .reset_index(name="count")
    )

    return {
        "available": True,
        "bucket_counts": [
            {"bucket": str(row["bucket"]), "count": int(row["count"])}
            for _, row in bucket_counts.iterrows()
        ],
        "largest_quarter_groups": [
            {
                "release_quarter_key": str(row["release_quarter_key"]),
                "count": int(row["count"]),
            }
            for _, row in group_counts.head(12).iterrows()
        ],
    }


def _apply_pre_release_temporal_split(df: pd.DataFrame) -> dict:
    known = df[df["release_year"].notna() & df["release_quarter"].notna()].copy()
    known["quarter_index"] = known["release_year"].astype(int) * 10 + known["release_quarter"].astype(int)
    unique_quarters = sorted(known["quarter_index"].dropna().unique().tolist())

    if not unique_quarters:
        df["split_pre_release"] = "unknown"
        return {"train_quarters": 0, "val_quarters": 0, "test_quarters": 0}

    total = len(unique_quarters)
    train_n = max(1, int(total * SPLIT_RATIOS["train"]))
    val_n = max(1, int(total * SPLIT_RATIOS["val"]))
    if train_n + val_n >= total:
        val_n = 1
        train_n = max(1, total - 2)
    test_n = max(1, total - train_n - val_n)
    if train_n + val_n + test_n > total:
        test_n = total - train_n - val_n

    train_quarters = set(unique_quarters[:train_n])
    val_quarters = set(unique_quarters[train_n : train_n + val_n])
    test_quarters = set(unique_quarters[train_n + val_n :])

    df["split_pre_release"] = "unknown"
    df.loc[known.index, "split_pre_release"] = "test"
    df.loc[known.index[known["quarter_index"].isin(train_quarters)], "split_pre_release"] = "train"
    df.loc[known.index[known["quarter_index"].isin(val_quarters)], "split_pre_release"] = "val"
    df.loc[known.index[known["quarter_index"].isin(test_quarters)], "split_pre_release"] = "test"

    split_counts = (
        df["split_pre_release"].value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
    )

    return {
        "train_quarters": len(train_quarters),
        "val_quarters": len(val_quarters),
        "test_quarters": len(test_quarters),
        "split_counts": [
            {"split": str(row["split"]), "count": int(row["count"])}
            for _, row in split_counts.iterrows()
        ],
    }


def _apply_unknown_split_policy(df: pd.DataFrame) -> dict:
    df["split_pre_release_effective"] = df["split_pre_release"]
    if UNKNOWN_SPLIT_POLICY == "exclude_unknown_from_model_splits":
        df.loc[df["split_pre_release"] == "unknown", "split_pre_release_effective"] = "holdout_unknown"
    df["is_model_split"] = df["split_pre_release_effective"].isin(["train", "val", "test"])

    effective_counts = (
        df["split_pre_release_effective"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
    )
    model_split_counts = (
        df[df["is_model_split"]]["split_pre_release_effective"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
    )

    return {
        "policy": UNKNOWN_SPLIT_POLICY,
        "excluded_unknown_count": int((df["split_pre_release"] == "unknown").sum()),
        "effective_split_counts": [
            {"split": str(row["split"]), "count": int(row["count"])}
            for _, row in effective_counts.iterrows()
        ],
        "model_split_counts": [
            {"split": str(row["split"]), "count": int(row["count"])}
            for _, row in model_split_counts.iterrows()
        ],
    }


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


def _write_target_summary(summary: dict) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Target Engineering Summary",
        "",
        f"- Rule version: `{summary['rule_version']}`",
        f"- Input file: `{summary['input_file']}`",
        "",
        "## Popularity Quarter Target",
        "",
    ]

    for item in summary["popularity_quarter_target"].get("bucket_counts", []):
        lines.append(f"- `{item['bucket']}`: {item['count']}")

    lines.extend(["", "## Temporal Pre-release Split", ""])
    split = summary["pre_release_split"]
    lines.append(f"- Train quarters: `{split['train_quarters']}`")
    lines.append(f"- Val quarters: `{split['val_quarters']}`")
    lines.append(f"- Test quarters: `{split['test_quarters']}`")
    for item in split.get("split_counts", []):
        lines.append(f"- `{item['split']}` rows: {item['count']}")

    lines.extend(["", "## Unknown Split Policy", ""])
    policy = summary["unknown_split_policy"]
    lines.append(f"- Policy: `{policy['policy']}`")
    lines.append(f"- Excluded unknown rows from model splits: `{policy['excluded_unknown_count']}`")
    for item in policy.get("effective_split_counts", []):
        lines.append(f"- Effective `{item['split']}` rows: {item['count']}")

    TARGET_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    input_file = _latest_interim_csv()
    df = pd.read_csv(input_file)

    non_negative_stats = _enforce_non_negative(df)
    clipping_stats = _clip_by_percentile(df)
    _derive_release_quarter(df)
    popularity_quarter_target = _add_popularity_quarter_target(df)
    pre_release_split = _apply_pre_release_temporal_split(df)
    unknown_split_policy = _apply_unknown_split_policy(df)

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
        "popularity_quarter_target": popularity_quarter_target,
        "pre_release_split": pre_release_split,
        "unknown_split_policy": unknown_split_policy,
    }
    PROCESSED_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(meta)
    _write_target_summary(meta)

    print(f"Wrote {PROCESSED_CSV}")
    print(f"Wrote {PROCESSED_META}")
    print(f"Wrote {OUTLIER_SUMMARY_JSON}")
    print(f"Wrote {OUTLIER_SUMMARY_MD}")
    print(f"Wrote {TARGET_SUMMARY_JSON}")
    print(f"Wrote {TARGET_SUMMARY_MD}")


if __name__ == "__main__":
    main()
