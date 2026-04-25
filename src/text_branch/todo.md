# Text Branch TODO

## Goal
Build a reproducible text-only pipeline for anime synopsis embeddings and regression on `popularity` and `meanScore`.

## Current Status
- [x] Create folder structure (`src/text_branch`, `artifacts`, `reports`)
- [x] Create text preprocessor module
- [x] Create embedding generator module
- [x] Configure CPU-safe embedding settings (`device: auto`)
- [x] Build embedding export script (all splits)
- [x] Build text-only baseline model training script
- [x] Generate evaluation metrics report
- [x] Verify split integrity and leakage checks
- [x] Document reproducibility metadata
- [x] Run full integration test (full dataset, no sample cap)

## Step-by-Step Plan

### 1) Environment Setup
- [x] Install required packages in your active environment

Command:
```bash
C:/Users/User/anaconda3/python.exe -m pip install sentence-transformers pandas numpy scikit-learn scipy pyarrow pyyaml
```

### 2) Embedding Export (Train/Val/Test)
- [x] Create `src/text_branch/run_text_embedding_pipeline.py`
- [x] Read split files:
  - `data/processed/anilist_anime_multimodal_input_train.csv`
  - `data/processed/anilist_anime_multimodal_input_val.csv`
  - `data/processed/anilist_anime_multimodal_input_test.csv`
- [x] Clean `description` text with `TextPreprocessor`
- [x] Generate embeddings with `EmbeddingGenerator`
- [x] Save artifacts:
  - `artifacts/text_embeddings_train.parquet`
  - `artifacts/text_embeddings_val.parquet`
  - `artifacts/text_embeddings_test.parquet`

### 3) Text-Only Baseline Modeling
- [x] Create `src/text_branch/baseline_model.py`
- [x] Train baseline regressor for `popularity`
- [x] Train baseline regressor for `meanScore`
- [ ] Start with `Ridge` (simple + stable), then optionally compare `MLPRegressor`

### 4) Evaluation and Reporting
- [x] Compute metrics on validation and test:
  - `MAE`
  - `RMSE`
  - `Spearman`
- [x] Save report: `reports/text_branch_metrics.json`

### 5) Reproducibility Checklist
- [x] Save model metadata (`model_name`, `embedding_dim`)
- [x] Record random seed in report/config
- [x] Keep split usage fixed (`train`, `val`, `test` only)
- [x] Keep `holdout_unknown` excluded from training
- [x] Add package version snapshot to metrics report

### 6) Quality Checks
- [x] Verify row counts before and after text cleaning
- [ ] Check null/empty description handling behavior
- [x] Confirm embedding dimensionality is consistent
- [x] Confirm no ID overlap across splits

## What To Do Next (Now)
- [x] Run full embeddings (remove `--sample-size`) for train/val/test
- [x] Re-run `baseline_model.py` on full artifacts
- [x] Add package-version metadata to `reports/text_branch_metrics.json`
- [ ] Add MLP comparison run (optional)

## Latest Run Snapshot
- Full embedding pipeline completed (train/val/test)
  - train encoded: 12783/13376 (95.57%)
  - val encoded: 2637/2918 (90.37%)
  - test encoded: 2808/3087 (90.96%)
- Baseline metrics generated: `reports/text_branch_metrics.json`
  - popularity: val (MAE 20462.82, RMSE 42125.67, Spearman 0.5509), test (MAE 17946.53, RMSE 34055.32, Spearman 0.5408)
  - meanScore: val (MAE 9.81, RMSE 11.93, Spearman 0.2886), test (MAE 10.94, RMSE 13.12, Spearman 0.2152)
- Findings summary document added: `reports/text_branch_experiment_findings.md`

## Notes
- Local CPU runtime for embeddings is workable on your setup (roughly a few minutes for train+val+test with MiniLM).
- Use Colab only if you move to larger models or heavy fine-tuning.
