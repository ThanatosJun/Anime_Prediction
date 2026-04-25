# Text Branch Experiment Findings

Date: 2026-04-25
Owner: Text branch
Scope: Pre-release text embedding and text-only baseline for popularity and meanScore regression

## 1. Objective
Provide reusable text embedding artifacts from anime synopsis (description) for downstream fusion with image and other modalities.

## 2. Pipeline Summary
Input source files:
- data/processed/anilist_anime_multimodal_input_train.csv
- data/processed/anilist_anime_multimodal_input_val.csv
- data/processed/anilist_anime_multimodal_input_test.csv

Text processing and embedding:
- Text field: description
- Cleaning: lowercase, URL removal, whitespace normalization, min length 10, max length 512
- Model: sentence-transformers/all-MiniLM-L6-v2
- Embedding size: 384
- Device: CPU
- Batch size: 16
- Random seed: 42

Produced artifacts:
- artifacts/text_embeddings_train.parquet
- artifacts/text_embeddings_val.parquet
- artifacts/text_embeddings_test.parquet

Supporting reports:
- reports/text_embedding_pipeline_summary.json
- reports/text_branch_metrics.json

## 3. Data Retention After Cleaning
- Train: 12783 / 13376 encoded (95.57%)
- Val: 2637 / 2918 encoded (90.37%)
- Test: 2808 / 3087 encoded (90.96%)

Interpretation:
- Most rows have usable text.
- A minority is dropped due to missing/invalid text after cleaning.
- Fusion stage must handle missing-text rows explicitly.

## 4. Baseline Model and Metrics
Baseline model:
- Ridge regression, alpha = 1.0
- Trained separately for each target using text embeddings only

Target: popularity
- Validation: MAE 20462.82, RMSE 42125.67, Spearman 0.5509
- Test: MAE 17946.53, RMSE 34055.32, Spearman 0.5408

Target: meanScore
- Validation: MAE 9.81, RMSE 11.93, Spearman 0.2886
- Test: MAE 10.94, RMSE 13.12, Spearman 0.2152

Interpretation:
- Text embeddings contain meaningful ranking signal for popularity.
- Text-only signal for meanScore exists but is weaker.
- This supports using text features in multimodal fusion, especially for popularity.

## 5. Split Integrity and Leakage Checks
ID overlap check across splits:
- train vs val: 0
- train vs test: 0
- val vs test: 0

Conclusion:
- No cross-split ID leakage detected in embedding artifacts.

## 6. Handoff Contract for Fusion Teammates
Use these as text features in fusion pipeline:
- emb_000 through emb_383 (384-dimensional vector)

Use these as join and supervision fields:
- id as join key
- split column for split-aligned usage
- popularity and meanScore as target references when needed

Recommended fusion practice:
- Keep split boundaries fixed by file (train with train, val with val, test with test)
- Define explicit policy for missing text rows (intersection, imputation, or missing-modality handling)

## 7. Reproducibility Snapshot
Package versions used in baseline report:
- python 3.13.5
- numpy 2.1.3
- pandas 2.2.3
- scipy 1.15.3
- scikit-learn 1.6.1
- pyarrow 19.0.0

Commands used:
- C:/Users/User/anaconda3/python.exe src/text_branch/run_text_embedding_pipeline.py --splits train val test
- C:/Users/User/anaconda3/python.exe src/text_branch/baseline_model.py

## 8. Recommended Next Optional Step
Run one additional text baseline (for example MLPRegressor) and compare against Ridge to provide a stronger text-branch benchmark for paper reporting.
