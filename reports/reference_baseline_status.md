# Reference Baseline Status

更新日期：2026-05-12

本文件記錄 `docs/baseline_reference_implementation_plan.md` 目前已實作與已跑通的 reference baseline 範圍。注意：這裡只記錄「文獻/外部比較 baseline」，不包含本專案自己的 ablation baseline。

## Summary

目前完成的是：

1. `0. Lowest Reference / 最低地板`
2. `1.1 Metadata-only Classical ML`

尚未完成的是：

1. `1.3 Text-only Baseline`
2. `1.4 Image-only Baseline`
3. `2.1 Anime Domain Deep Fusion`
4. `2.2 Cross-modal Transformer Fusion`
5. `2.3 Retrieval / RAG Competitive Baseline`

換句話說：目前不是完成 Armenta-Segura & Sidorov 2025 那條 competitive baseline，而是先完成 reference baseline map 裡的 foundation/classical 與 feature-concat 起始段。

## Plan Mapping

| Plan route | Baseline IDs | Status | Notes |
|---|---|---|---|
| `0. Lowest Reference / 最低地板` | `F0-Mean`, `F0-Ridge-Meta` | done | Mean predictor 與 Ridge metadata baseline 已跑 `popularity`、`meanScore` |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | done as adaptation | 對應 Lo & Syu 2023 的 pre-broadcast metadata + classical ML 參考；RF 有原文方法支撐，Gradient Boosting 是本專案延伸強 tabular baseline，不是原文模型 |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | done as adaptation | 已補 `docs/reference_baselines/f2_feature_concat_plan.md`；metadata + text embedding + image embedding concat 架構已用真實 embeddings 與 XGBoost 跑通 `popularity`、`meanScore` |
| `1.3 Text-only Baseline` | not implemented | todo | 尚未建立 TF-IDF / text embedding only reference runner |
| `1.4 Image-only Baseline` | not implemented | todo | 尚未建立 cover/banner image-only reference runner |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP` | scaffold only | config 中有 disabled scaffold；需 text/image embeddings 後才能跑 |
| `2.2 Cross-modal Transformer Fusion` | not implemented | todo | 尚未實作 CTNN / cross-attention baseline |
| `2.3 Retrieval / RAG Competitive Baseline` | not implemented | todo | 尚未實作 SKAPP-inspired reference runner；現有 RAG 對照較偏 project ablation |

## Implemented Code

Reference baseline code:

```text
src/reference_baseline_branch/
├── configs/reference_baselines.yaml
├── sklearn_models.py
├── run_reference_baselines.py
└── README.md
```

Shared experiment utilities:

```text
src/experiment_common/
├── features.py
├── metrics.py
└── README.md
```

Project ablation scaffold, separate from reference baselines:

```text
src/ablation_branch/
├── configs/ablation_baselines.yaml
├── run_ablation_baselines.py
└── README.md
```

## Latest Full Reference Run

Latest full run before directory split:

```text
.exp/baseline/results/05/baseline_results.csv
```

Important note: `.exp/` is ignored and should be treated as local experiment output. This markdown file is the tracked status record.

Tracked result summary:

```text
reports/reference_baseline_results.csv
reports/reference_baseline_runs.md
```

Latest F2 feature-concat run:

```text
.exp/baseline/results/14/baseline_results.csv
```

Embedding coverage used by strict intersection:

| split | metadata ids | text ids in split | image ids in split | common ids used |
|---|---:|---:|---:|---:|
| train | 9583 | 9205 | 9583 | 9205 |
| val | 2918 | 2637 | 2918 | 2637 |
| test | 3087 | 2808 | 3087 | 2808 |

## Paper Reference Check

### Claim Boundary Table

| baseline_id | reproduction_level | paper_supported_component | project_adaptation_component | claim_allowed | claim_not_allowed |
|---|---|---|---|---|---|
| `F0-Mean` | common | Common lowest-reference predictor; no specific paper reproduction claimed. | Predict train-set target mean for every validation/test sample. | Used as a lowest-reference baseline. | Do not cite this as reproducing any paper. |
| `F0-Ridge-Meta` | common | Common linear baseline; no specific paper reproduction claimed. | Ridge regression on AniList pre-release metadata using project temporal split. | Used as a simple linear metadata baseline. | Do not cite this as reproducing Lo & Syu 2023 or another paper. |
| `F1-RF-Meta` | adapted | Lo & Syu 2023 supports pre-broadcast entertainment metadata with classical ML and includes Random Forest. | Anime-domain regression on popularity/meanScore with AniList pre-release metadata and temporal split. | Metadata-only Random Forest baseline adapted from Lo & Syu 2023. | Do not claim exact reproduction; the original task is Japanese TV drama high/low rating classification with cross-validation. |
| `F1-GB-Meta` | adapted | Lo & Syu 2023 supports the metadata-only classical ML route, but not Gradient Boosting specifically. | Stronger sklearn Gradient Boosting tabular baseline on the same AniList metadata feature set. | Gradient-boosting extension of the Lo & Syu 2023 metadata-only classical ML route. | Do not claim this model appears in Lo & Syu 2023. |
| `F2-XGB-Concat` | adapted | Chen et al. 2019 and Jeong et al. 2024 support visual/textual feature fusion with XGBoost-style classical regressors. | AniList metadata + project text embeddings + project image embeddings concatenated for XGBoost regression. | Feature-concat XGBoost baseline adapted from visual-textual popularity prediction literature. | Do not claim same dataset, encoders, or exact feature extraction as the cited papers. |
| `C1-Armenta-MLP` | adapted | Armenta-Segura & Sidorov 2025 supports anime-domain multimodal popularity prediction using text/image branches and deep fusion. | Project metadata + text embeddings + image embeddings with an sklearn MLP fusion head. | Anime-domain deep fusion baseline adapted from Armenta-Segura & Sidorov 2025. | Do not claim numerical comparability or exact reproduction; model branches, dataset, and target definitions differ. |

### `F0-Mean` / `F0-Ridge-Meta`

These are common lowest-reference baselines. They do not reproduce a specific paper.

### `F1-RF-Meta`

This is a project adaptation of Lo & Syu (2023), not an exact reproduction.

What is supported by the paper:

- Entertainment-domain pre-broadcast metadata can be used for prediction.
- The paper uses Japanese prime-time TV drama metadata such as broadcast year, season, time slot, station, genre, original/sequel status, screenwriter, and cast.
- The paper evaluates classical machine learning classifiers including Random Forest.
- The paper uses metadata classification for high/low rating groups, not anime regression.

What this project changes:

- Domain changes from Japanese TV dramas to anime.
- Target changes from high/low rating classification to regression on `popularity` and `meanScore`.
- Split policy changes from the paper's cross-validation setting to this project's temporal train/val/test split.
- Metadata fields are AniList-equivalent pre-release fields, not the exact drama metadata fields.

### `F1-GB-Meta`

This is not directly from Lo & Syu (2023). It is a stronger classical tabular extension under the same `Metadata-only Classical ML` route. It should be reported as "Lo & Syu 2023 + gradient boosting adaptation", not as a direct reproduction.

### Completed Results Snapshot

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho | status |
|---|---:|---:|---:|---:|---|
| `F0-Mean` | popularity | 15034.3970 | -0.1368 | 0.0000 | ok |
| `F0-Mean` | meanScore | 10.4094 | -0.3536 | 0.0000 | ok |
| `F0-Ridge-Meta` | popularity | 15222.9838 | -2.2072 | 0.7995 | ok |
| `F0-Ridge-Meta` | meanScore | 8.5854 | 0.0075 | 0.5084 | ok |
| `F1-RF-Meta` | popularity | 8590.0532 | 0.5811 | 0.8466 | ok |
| `F1-RF-Meta` | meanScore | 7.9541 | 0.1298 | 0.5836 | ok |
| `F1-GB-Meta` | popularity | 8917.8924 | 0.4951 | 0.8367 | ok |
| `F1-GB-Meta` | meanScore | 8.7243 | -0.0269 | 0.5380 | ok |
| `F2-XGB-Concat` | popularity | 9588.2590 | 0.5194 | 0.8575 | ok |
| `F2-XGB-Concat` | meanScore | 8.3391 | 0.0193 | 0.5292 | ok |

### F2 Architecture Smoke Test

On 2026-05-12, the feature-concat path was tested with synthetic text/image embeddings:

```text
python -m src.reference_baseline_branch.run_reference_baselines --config .exp/f2_feature_concat_smoke/f2_smoke_config.yaml --baseline F2-LGBM-Concat-Smoke --target popularity
```

Result:

```text
status = ok
n_train = 9583
n_val = 2918
n_test = 3087
n_features = 159
```

This confirmed the runner could execute metadata + text embedding parquet + image embedding parquet through ID alignment, feature concatenation, model fitting, metrics, and result output before the real embeddings were available. The smoke result is not reportable as a real baseline because the embeddings were synthetic and the smoke model used LightGBM, not XGBoost.

### F2 Real Embedding Run

On 2026-05-12, real text/image embeddings were placed under the configured directories and `xgboost` was installed in the current Python environment.

Command:

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline F2-XGB-Concat
```

Result:

```text
status = ok
n_train = 9205
n_val = 2637
n_test = 2808
n_features = 1559
```

## Reproduction Commands

Run all enabled reference baselines:

```bash
python -m src.reference_baseline_branch.run_reference_baselines
```

Run a single baseline:

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline F1-RF-Meta --target popularity
```

Check ablation scaffold separately:

```bash
python -m src.ablation_branch.run_ablation_baselines
```

## Next Required Step

To move from completed foundation/classical baselines to the first competitive route:

1. Review whether `F2-XGB-Concat` should use strict intersection as final reporting policy or add a separately named missing-modality ablation.
2. Enable and run `C1-Armenta-MLP`.
3. Compare `C1-Armenta-MLP` against `F2-XGB-Concat`.

Only after step 3 can we say the `2.1 Anime Domain Deep Fusion` route has been implemented and compared against the feature-concat floor.
