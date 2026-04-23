# Data Processing Record for Paper Writing

This document records the data processing decisions in a paper-ready format.
It focuses on reproducibility, rationale, and modeling impact.

## 1) Dataset Scope and Snapshot Control

- **Domain:** Anime pre-release prediction.
- **Raw source snapshot:** `data/raw/anilist_anime_data_complete.pkl` and `data/raw/anilist_anime_data_complete.csv`.
- **Snapshot fingerprint:** `data/raw/raw_manifest.json` (sha256 + file size).
- **Motivation:** prevent silent dataset drift when upstream source is updated.

## 2) Processing Stages and Artifacts

### Stage A: Baseline EDA (`scripts/run_baseline_eda.py`)
- **Purpose:** descriptive profiling before rule decisions.
- **Outputs:** `data/eda/baseline_eda_summary.json/.md`.
- **Key stats tracked:** missing rate, numeric distribution, IQR outlier bounds.

### Stage B: Decision EDA (`scripts/run_decision_eda.py`)
- **Purpose:** convert descriptive stats into actionable policies.
- **Outputs:** `data/eda/decision_eda_summary.json/.md`.
- **Decision signals:**
  - missing-value policy recommendation (`drop/fill/keep`) by missing ratio
  - outlier policy recommendation (`clip/winsorize/retain`) by outlier ratio
  - correlation profile to target columns
  - grouped impact summary (`format` -> `popularity`)

### Stage C: Interim Dataset (`scripts/build_interim_dataset.py`)
- **Output:** `data/interim/anilist_anime_data_interim_YYYYMMDD.csv` + metadata json.
- **Current rule version:** `decision_eda_v1` (interim cleaning layer).
- **Operations:**
  - keep model-relevant columns only
  - enforce numeric dtypes
  - deduplicate by `id`
  - apply missing-value fills via explicit policy mapping (`MISSING_RULES`)

### Stage D: Processed Dataset (`scripts/build_processed_dataset.py`)
- **Output:** `data/processed/anilist_anime_data_processed_v1.csv` + metadata json.
- **Current rule version:** `decision_eda_v2` (processed + target layer).
- **Operations:**
  - non-negative constraints for key numeric features
  - percentile clipping with explicit `CLIP_COLUMNS`
  - quarter-normalized popularity target engineering
  - chronological pre-release split assignment (`train/val/test/unknown`)
  - unknown split policy: move `unknown` to `holdout_unknown` (excluded from model splits)

### Stage E: RQ-oriented EDA (`scripts/run_rq_eda.py`)
- **Purpose:** provide paper-level evidence tied to RQ concerns.
- **Outputs:** `data/eda/rq_eda_summary.json/.md`.
- **Current evidence tracked:**
  - snapshot mitigation proxy (`corr(release_year, popularity_raw)` vs `corr(release_year, popularity_quarter_pct)`)
  - absolute correlation reduction after quarter normalization
  - RQ1 proxy readiness (metadata/relation coverage + effective split distribution)
  - popularity class balance by model split
  - RQ2 proxy readiness (text/image/trailer source coverage)
  - multimodal source coverage by split (train/val/test/holdout_unknown)

## 3) Explicit Rules Used in Current Version

### 3.1 Missing-value rules (interim)
- `episodes`: format median, fallback global median
- `duration`: format median, fallback global median
- `averageScore`: fill from `meanScore`, fallback global median
- `seasonYear`: fill from `startDate_year`
- `title_english`: fill from `title_romaji`

### 3.2 Outlier rules (processed)
- `episodes`: clip at P1-P99
- `duration`: clip at P1-P99
- `averageScore`: clip at P0.5-P99.5
- `meanScore`: clip at P0.5-P99.5
- `popularity`: clip at P1-P99
- `favourites`: clip at P1-P99
- `trending`: clip at P1-P95

## 4) Popularity Target Engineering for Snapshot Mitigation

### 4.1 Quarter key construction
- Build `release_year` from `seasonYear` (fallback `startDate_year`).
- Build `release_quarter` from:
  - `season` mapping (`WINTER=1`, `SPRING=2`, `SUMMER=3`, `FALL=4`)
  - fallback from `startDate_month` if `season` is missing.
- Combine as `release_quarter_key` (e.g., `2021Q3`).

### 4.2 Relative popularity target
- Compute within-quarter percentile rank:
  - `popularity_quarter_pct = rank(popularity within release_quarter_key, pct=True)`
- Bucket definition:
  - `cold_0_25` : 0-25%
  - `warm_25_50` : 25-50%
  - `hot_50_75` : 50-75%
  - `top_75_100` : 75-100%

### 4.3 Rationale
- Raw popularity is cumulative and time-biased (snapshot issue).
- Relative ranking within same quarter is more comparable for pre-release settings.

## 5) Pre-release Temporal Split Protocol

- Build quarter index: `quarter_index = release_year * 10 + release_quarter`.
- Sort unique quarters chronologically.
- Assign quarter buckets by ratio:
  - train: 70%
  - val: 15%
  - test: 15%
- Map each sample to `split_pre_release` via its quarter index.
- Missing quarter info is labeled as `unknown`.
- Apply unknown policy:
  - `unknown` -> `holdout_unknown` in `split_pre_release_effective`
  - exclude `holdout_unknown` from train/val/test model splits

## 6) Reproducibility Evidence

- Raw snapshot integrity: `data/raw/raw_manifest.json`
- Rule version trace:
  - interim metadata: `rule_version`, `applied_missing_rules`
  - processed metadata: `rule_version`, `clip_config`, `popularity_quarter_target`, `pre_release_split`
- Decision evidence:
  - `data/eda/decision_eda_summary.*`
  - `data/eda/target_engineering_summary.*`
  - `data/eda/outlier_handling_summary.*`
  - `data/eda/rq_eda_summary.*`

## 7) Current Limitations and Paper Notes

- Current split still contains `unknown` bucket for missing temporal fields.
- Relative popularity labels reduce snapshot bias but do not remove all lifecycle effects.
- Pipeline currently focuses on tabular preprocessing; multimodal embedding pipeline is a later stage.
- For the paper, report both:
  - raw metric behavior (before normalization)
  - quarter-normalized behavior (after target engineering)

## 8) Minimal Commands to Reproduce

```bash
python scripts/generate_raw_manifest.py
python scripts/run_baseline_eda.py
python scripts/run_decision_eda.py
python scripts/build_interim_dataset.py
python scripts/build_processed_dataset.py
```
