# Data Processing Record for Paper Writing

This document records the data processing and research-scope decisions in a paper-ready format.
It focuses on reproducibility, rationale, modeling impact, and project scope alignment.

## 0) Core Objective and Scope Definition

- **Primary objective:** predict post-release outcomes (popularity and mean score) using **pre-release** multimodal signals, including image, text description, and metadata.
- **Target protocol (current decision):** both primary targets are treated as regression tasks; no fixed `Day7` target constraint in the current scope.
- **Pre-release definition:** the stage before a title is officially broadcast/released/sold.
- **Current project phase:** data pipeline and evidence layer construction (not final model training report yet).

## 0.1) Domain Evaluation and Final Domain Selection

The team evaluated multiple candidate domains and selected **Anime** as the main domain.

- **A. Anime (selected)**
  - Main dataset used in this repository: AniList (`20,324` rows snapshot).
  - Primary targets: popularity and mean score.
  - Core risk identified: snapshot bias caused by cumulative popularity over time.
- **B. Airbnb pricing (not selected)**
  - Short-term prices are highly event/date-driven.
  - Static snapshot features are insufficient for reliable pre-event fluctuation modeling.
- **C. Other explored domains (not selected)**
  - Movie/Game: download/adoption labels are difficult to collect consistently, and/or still contain snapshot-style drift.
  - YouTube: standardized public metrics provide limited room for meaningful new predictive signal design.

## 0.2) Planned Modeling Architecture (Research Design Layer)

The intended downstream modeling design is a fusion setup:

- **Image branch:** Swin Transformer (or ResNet-50 baseline) for cover/visual features.
- **Text branch:** Transformer-based encoder (e.g., GPT-2 family style embedding pipeline) for synopsis/description semantics.
- **Metadata branch:** structured features such as genres, episodes, studio, relation context, and cast-related signals.
- **Retrieval augmentation (RQ1-related):** use relation/company/sequel links as retrieval context to support prediction.

> Note: this document tracks what is already implemented in the pipeline and what is defined as the next experiment layer. The architecture above is a research design target, while the current repository mainly completes data and evidence readiness.

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

### Stage E: Multimodal Input Export (`scripts/export_multimodal_inputs.py`)
- **Purpose:** preserve text/image/trailer fields for multimodal modeling while keeping split-aligned targets.
- **Outputs:**
  - `data/processed/anilist_anime_multimodal_input_v1.csv`
  - `data/processed/anilist_anime_multimodal_input_{train|val|test|holdout_unknown}.csv`
  - `data/eda/multimodal_input_summary.json/.md`
- **Current evidence tracked:**
  - feature contract (join key, target columns, raw modality columns)
  - modality availability flags and ratios
  - physical split row counts

### Stage F: RQ-oriented EDA (`scripts/run_rq_eda.py`)
- **Purpose:** provide paper-level evidence tied to RQ concerns.
- **Outputs:** `data/eda/rq_eda_summary.json/.md`.
- **Current evidence tracked:**
  - snapshot mitigation proxy (`corr(release_year, popularity_raw)` vs `corr(release_year, popularity_quarter_pct)`)
  - absolute correlation reduction after quarter normalization
  - RQ1 proxy readiness (metadata/relation coverage + effective split distribution)
  - popularity class balance by model split
  - RQ2 proxy readiness (text/image/trailer source coverage)
  - multimodal source coverage by split (train/val/test/holdout_unknown)
  - statistical test layer:
    - permutation test for split bucket balance shift
    - permutation tests for multimodal coverage gap across splits
    - bootstrap CI for snapshot-correlation reduction

### Stage G: RQ Figure Generation (`scripts/run_rq_eda_plots.py`)
- **Purpose:** convert RQ EDA indicators into direct paper figures.
- **Outputs:** `data/eda/figures/*.png` + `data/eda/figures/rq_figure_notes.md`.
- **Current figure set:**
  - snapshot bias proxy (absolute correlation before/after quarter normalization)
  - popularity bucket balance by split
  - multimodal coverage by split

### Stage H: Holdout Unknown Diagnostic (`scripts/run_holdout_unknown_diagnostic.py`)
- **Purpose:** quantify the risk profile of excluded temporal-unknown samples.
- **Outputs:** `data/eda/holdout_unknown_diagnostic.json/.md`.
- **Current evidence tracked:**
  - holdout size and ratio in the full dataset
  - temporal field missing profile
  - distribution gaps vs model-split population for key targets/features

### Stage I: Column Lineage Report (`scripts/run_column_lineage_report.py`)
- **Purpose:** provide explicit raw->interim->processed->multimodal column-level transformation evidence.
- **Outputs:** `data/eda/column_lineage_summary.json/.md`.
- **Current evidence tracked:**
  - stage-wise column counts
  - keep/drop/add sets across each stage
  - derived-column origin mapping to transformation functions
  - multimodal reintroduced fields and availability-flag derivation reasons

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

### 4.2 Relative popularity target (auxiliary feature / diagnostic layer)
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
- Sort quarter groups chronologically.
- Assign split cut points by **cumulative row count** (not quarter count):
  - train target: 70%
  - val target: 15%
  - test target: 15%
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
  - `data/eda/multimodal_input_summary.*`
  - `data/eda/decision_eda_summary.*`
  - `data/eda/target_engineering_summary.*`
  - `data/eda/outlier_handling_summary.*`
  - `data/eda/rq_eda_summary.*`
  - `data/eda/holdout_unknown_diagnostic.*`
  - `data/eda/column_lineage_summary.*`

## 7) Current Limitations and Paper Notes

- Current split still contains `unknown` bucket for missing temporal fields.
- Relative popularity labels reduce snapshot bias but do not remove all lifecycle effects.
- Pipeline currently focuses on tabular preprocessing; multimodal embedding pipeline is a later stage.
- For the paper, report both:
  - raw metric behavior (before normalization)
  - quarter-normalized behavior (after target engineering)

## 7.1) Explicit RQ Mapping and Evaluation Plan

- **RQ1:** whether retrieval-based augmentation improves regression performance on popularity and mean score targets.
- **RQ2:** whether transformer-based image semantics provide measurable gain beyond simple tag-style features.
- **Interpretability plan:**
  - use SHAP for metadata contribution analysis,
  - use ablation study (`tabular+text`, `+image`, `+retrieval`) for incremental gain verification.

## 7.2) Scope Boundary and Response to Prior Concerns

- The current stage performs **pre-release** prediction preparation only; it does not include real-time post-release social diffusion signals.
- Snapshot control is handled by quarter-relative popularity engineering and chronological split protocol.
- Relation/studio/cast and multimodal availability are preserved as research-readiness evidence in EDA and lineage outputs.
- The popularity bucket fields (`popularity_quarter_pct`, `popularity_quarter_bucket`) are used as auxiliary diagnostics and control features, not as the final statement of target definition.

## 8) Minimal Commands to Reproduce

```bash
python scripts/generate_raw_manifest.py
python scripts/run_baseline_eda.py
python scripts/run_decision_eda.py
python scripts/build_interim_dataset.py
python scripts/build_processed_dataset.py
python scripts/run_rq_eda.py
python scripts/run_rq_eda_plots.py
python scripts/run_holdout_unknown_diagnostic.py
python scripts/run_column_lineage_report.py
```
