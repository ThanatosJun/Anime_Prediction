# Data Pipeline Handoff Guide

This guide is for teammates who need to rerun or modify the AniList dataset pipeline.

## 1) End-to-End Rebuild Order

Run from repository root:

```bash
python scripts/generate_raw_manifest.py
python scripts/run_baseline_eda.py
python scripts/run_decision_eda.py
python scripts/build_interim_dataset.py
python scripts/build_processed_dataset.py
python scripts/export_multimodal_inputs.py
python scripts/run_rq_eda.py
python scripts/run_rq_eda_plots.py
python scripts/run_holdout_unknown_diagnostic.py
python scripts/run_column_lineage_report.py
```

## 2) What Each Output Means

- `data/raw/raw_manifest.json`
  - Frozen fingerprint (size + sha256) for canonical raw files.
- `data/eda/baseline_eda_summary.*`
  - Descriptive quality profile (missing rates, distributions, IQR bounds).
- `data/eda/decision_eda_summary.*`
  - Actionable recommendations for missing-value and outlier strategies.
- `data/interim/anilist_anime_data_interim_YYYYMMDD_meta.json`
  - Applied cleaning rule version and missing-value policy.
- `data/processed/anilist_anime_data_processed_v1_meta.json`
  - Applied outlier thresholds, clip configuration, and rule version.
- `data/processed/anilist_anime_multimodal_input_v1.csv` and split files
  - Multimodal-ready feature contract export with `train/val/test/holdout_unknown` physical files.
- `data/eda/multimodal_input_summary.*`
  - Feature contract summary and modality coverage ratios for downstream model stages.
- `data/eda/target_engineering_summary.*`
  - Quarter-normalized popularity label distribution and pre-release temporal split summary.
- `data/eda/rq_eda_summary.*`
  - RQ-aligned readiness signals and snapshot-control evidence.
  - Also includes class balance by split, split-wise multimodal coverage, and permutation/bootstrap statistical tests.
- `data/eda/figures/*.png`
  - Paper-ready figures for snapshot control, split balance, and multimodal coverage.
- `data/eda/figures/rq_figure_notes.md`
  - Figure-level interpretation notes for writing.
- `data/eda/holdout_unknown_diagnostic.*`
  - Holdout unknown composition and distribution-gap diagnostics.
- `data/eda/column_lineage_summary.*`
  - Raw->Interim->Processed column keep/drop/add lineage report.

## 2.1) Processed 6-CSV Mapping

- `data/processed/anilist_anime_data_processed_v1.csv`
  - Role: tabular master dataset (engineering contract for baseline and diagnostics)
  - Rows/Columns: `20324` / `38`
  - Usage: tabular modeling, full feature checks, split policy validation
- `data/processed/anilist_anime_multimodal_input_v1.csv`
  - Role: multimodal master dataset (text/image/trailer-ready contract)
  - Rows/Columns: `20324` / `21`
  - Usage: single entry point for multimodal branch experiments
- `data/processed/anilist_anime_multimodal_input_train.csv`
  - Role: physical train split export
  - Rows/Columns: `13376` / `21`
  - Usage: direct train loader input without runtime split filtering
- `data/processed/anilist_anime_multimodal_input_val.csv`
  - Role: physical validation split export
  - Rows/Columns: `2918` / `21`
  - Usage: hyperparameter tuning and early stopping
- `data/processed/anilist_anime_multimodal_input_test.csv`
  - Role: physical test split export
  - Rows/Columns: `3087` / `21`
  - Usage: final held-out performance reporting
- `data/processed/anilist_anime_multimodal_input_holdout_unknown.csv`
  - Role: temporal-unknown holdout export
  - Rows/Columns: `943` / `21`
  - Usage: risk diagnosis only; excluded from train/val/test model fitting

## 2.2) Why `val` and `test` counts are different

- Split policy is **chronological + quarter-block assignment** (`chronological_cumulative_rows`).
- The pipeline does **not** split a single quarter across different splits.
- Because quarter sizes are unequal, exact `15% / 15%` equality is not guaranteed.
- Current known-time rows: `19381` (`20324 - 943 holdout_unknown`).
- Current realized ratios on known-time rows:
  - train: `13376 / 19381` = `69.0%`
  - val: `2918 / 19381` = `15.1%`
  - test: `3087 / 19381` = `15.9%`
- This behavior is expected and preferable for temporal integrity (prevents quarter leakage).

## 3) Where to Change Rules

- Missing-value policy:
  - `scripts/build_interim_dataset.py`
  - Update `MISSING_RULES` and logic in `impute_missing_values()`.
- Outlier policy:
  - `scripts/build_processed_dataset.py`
  - Update `CLIP_COLUMNS` and logic in `_clip_by_percentile()`.
- Popularity target + temporal split policy:
  - `scripts/build_processed_dataset.py`
  - Update `_add_popularity_quarter_target()` and `_apply_pre_release_temporal_split()`.
  - Temporal split strategy is chronological + cumulative row ratio (70/15/15 target).
- Multimodal feature export contract:
  - `scripts/export_multimodal_inputs.py`
  - Update selected raw modality columns and exported split file contract.
- Recommendation logic:
  - `scripts/run_decision_eda.py`
  - Update `_missing_strategy()` and `_outlier_strategy()`.
- RQ-oriented evidence layer:
  - `scripts/run_rq_eda.py`
  - Update snapshot/retrieval/multimodal proxy metrics and statistical test settings.
- Holdout risk diagnostic:
  - `scripts/run_holdout_unknown_diagnostic.py`
  - Update temporal-missing and distribution-gap checks.
- Column lineage report:
  - `scripts/run_column_lineage_report.py`
  - Update stage-wise keep/drop/add interpretation for paper traceability.

## 4) Common Update Scenarios

### A. Raw dataset is refreshed by author
1. Replace local raw files in `data/raw` (`pkl/csv`).
2. Run `scripts/generate_raw_manifest.py`.
3. Run full rebuild sequence.
4. Verify decision and outlier summaries changed as expected.

### B. New field is needed in model features
1. Add field to `KEEP_COLUMNS` in `scripts/build_interim_dataset.py`.
2. Add field type handling in `NUMERIC_COLUMNS` if numeric.
3. Rebuild and inspect missing/outlier policy outputs.

### C. Outlier clipping feels too aggressive
1. Adjust quantiles in `CLIP_COLUMNS`.
2. Rebuild processed dataset.
3. Compare `outlier_handling_summary.*` before/after counts and bounds.

### D. Quarterly popularity class definitions need adjustment
1. Update bucket labels or thresholds in `_add_popularity_quarter_target()`.
2. Rebuild processed dataset.
3. Check `target_engineering_summary.*` for class distribution drift.

## 5) Verification Checklist Before Handoff

- Raw fingerprint exists and matches current files (`raw_manifest.json`).
- Decision summary exists (`decision_eda_summary.json/.md`).
- Interim metadata includes `rule_version` and `applied_missing_rules`.
- Processed metadata includes `rule_version` and `clip_config`.
- Processed metadata includes `popularity_quarter_target` and `pre_release_split`.
- Processed metadata includes `unknown_split_policy`.
- Multimodal input export files exist (full + split files).
- Multimodal input summary exists (`multimodal_input_summary.json/.md`).
- RQ summary exists (`rq_eda_summary.json/.md`).
- RQ figure files and figure notes exist under `data/eda/figures/`.
- Holdout diagnostic outputs exist (`holdout_unknown_diagnostic.json/.md`).
- Column lineage outputs exist (`column_lineage_summary.json/.md`).
- Pipeline runs without manual edits from a clean shell session.

