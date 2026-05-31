# F2 Feature-Concat Classical ML Plan

更新日期：2026-05-12

## 1. Route

本文件對應 `docs/baseline_reference_implementation_plan.md` 的：

```text
1.2 Feature-concat Classical ML
```

Baseline ID：

```text
F2-XGB-Concat
```

## 2. Reference Papers

主要參考：

1. Chen et al. 2019, `Social Media Popularity Prediction Based on Visual-Textual Features with XGBoost`
2. Jeong et al. 2024, `Enhancing Social Media Post Popularity Prediction with Visual Content`

## 3. Claim Boundary

Allowed claim:

> We implement a feature-concat classical ML baseline adapted from visual-textual popularity prediction literature.

Not allowed:

> We reproduce Chen et al. 2019 or Jeong et al. 2024.

原因：

- Dataset 不同：social media / Instagram post vs AniList anime.
- Feature extractors 不同：本專案使用既有 text/image embeddings，不重做原文的 visual/textual feature extraction。
- Target 不同：本專案預測 `popularity` 與 `meanScore`。
- Split 不同：本專案使用 temporal train/val/test split。

因此 `F2-XGB-Concat` 是 paper-inspired/adapted baseline，不是 exact reproduction。

## 4. Purpose

`F2-XGB-Concat` 是 competitive baselines 前的重要強地板，用來回答：

> 如果不做 deep fusion，只把 metadata、text embedding、image embedding 串接後交給強 tabular regressor，能做到哪裡？

這條 baseline 的作用：

1. 作為 `C1-Armenta-MLP` 的對照。
2. 判斷 deep fusion 是否真的比 simple feature concat 有價值。
3. 確認 text/image embeddings 是否在 classical ML 下也能提供訊號。

## 5. Inputs

Required inputs:

| Component | Source | Required |
|---|---|---|
| metadata | `data/fussion/post2000/fusion_meta_clean_{split}.csv` | yes |
| text embedding | `src/fussion_branch/embedding/text/text_embeddings_{split}.parquet` | yes |
| image embedding | `src/fussion_branch/embedding/image/image_embeddings_{split}.parquet` | yes |

Splits:

```text
train / val / test
```

Targets:

```text
popularity
meanScore
```

## 6. ID Alignment Policy

第一版採用 strict intersection：

```text
common_ids = metadata ids ∩ text embedding ids ∩ image embedding ids
```

理由：

1. F2 的定位是 full feature-concat baseline。
2. 若缺 text/image 時補零，容易混入 missing-modality handling 的 ablation 問題。
3. 補零策略應留給 project ablation，而不是 reference baseline 第一版。

每個 split 需記錄：

```text
n_metadata
n_text_embedding
n_image_embedding
n_common
coverage_ratio
```

若任一 embedding 檔案缺失：

```text
status = skipped
notes = missing embedding artifact
```

## 7. Feature Construction

Feature vector:

```text
X = concat(metadata_features, text_embedding, image_embedding)
```

Metadata encoder：

- fit on train only
- transform val/test
- numeric standardization uses train statistics
- categorical/multihot vocabularies use train only

Text/image embeddings：

- loaded from parquet
- no scaler in first version unless dimensions show instability
- aligned by `id`

## 8. Model Policy

Preferred model:

```text
XGBoost Regressor
```

Fallback policy:

| Situation | Action | Claim |
|---|---|---|
| `xgboost` installed | run `F2-XGB-Concat` | XGBoost feature-concat adaptation |
| `xgboost` missing | skip by default | no XGBoost result |
| team explicitly allows fallback | run `F2-GB-Concat` or `F2-HGB-Concat` | sklearn gradient-boosting feature-concat fallback, not XGBoost |

重要：fallback 結果不能寫成 `F2-XGB-Concat`。

## 9. Target Transform

Use existing reference baseline target policy:

| Target | Training transform | Report scale |
|---|---|---|
| `popularity` | `log1p(popularity)` | original scale + `log_MAE` |
| `meanScore` | raw | original scale |

## 10. Metrics

Use the shared reference baseline metrics:

```text
MAE
RMSE
R2
Spearman_rho
Pearson_r
log_MAE  # popularity only
```

## 11. Expected Output

Runner output:

```text
.exp/baseline/results/{run_id}/baseline_results.csv
.exp/baseline/results/{run_id}/baseline_summary.md
.exp/baseline/results/{run_id}/predictions/F2-XGB-Concat/{target}/{split}_predictions.csv
```

Tracked status update:

```text
reports/reference_baseline_status.md
```

## 12. Relationship To C1

| Baseline | Model type | Fusion style | Purpose |
|---|---|---|---|
| `F2-XGB-Concat` | classical ML | raw feature concat | strong tabular/multimodal floor |
| `C1-Armenta-MLP` | neural | learned MLP fusion | anime-domain deep fusion adaptation |

Interpretation:

- If `C1` does not beat `F2`, deep fusion may not be justified.
- If `F2` is close to `C1`, embeddings are useful but architecture gain is small.
- If `C1` clearly beats `F2`, deep fusion adds value beyond simple concatenation.

## 13. Implementation Checklist

1. Check whether text/image embeddings already exist under the configured paths.
2. If missing, generate or restore embeddings.
3. Confirm parquet schemas and embedding columns.
4. Run:

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline F2-XGB-Concat --target popularity
python -m src.reference_baseline_branch.run_reference_baselines --baseline F2-XGB-Concat --target meanScore
```

5. Update `reports/reference_baseline_status.md`.

## 14. Artifact Check

Checked on 2026-05-12:

| Artifact | Expected path | Current status |
|---|---|---|
| text embeddings | `src/fussion_branch/embedding/text/text_embeddings_{split}.parquet` | present; 384 dims; columns `id`, `emb_000...` |
| image embeddings | `src/fussion_branch/embedding/image/image_embeddings_{split}.parquet` | present; 1024 dims; columns `id`, `img_0...` |
| RAG none features | `src/fussion_branch/RAG/return/none/rag_features_{split}.parquet` | present, not required for F2 |
| fine-tuned image checkpoint | `results/01/best/` | present |
| local image files | `data/image/...` | not required for F2 once image embedding parquet is present |

Strict-intersection coverage:

| split | metadata ids | text ids in split | image ids in split | common ids used |
|---|---:|---:|---:|---:|
| train | 9583 | 9205 | 9583 | 9205 |
| val | 2918 | 2637 | 2918 | 2637 |
| test | 3087 | 2808 | 3087 | 2808 |

## 15. Architecture Smoke Test

Checked on 2026-05-12 with synthetic embeddings:

```text
.exp/f2_feature_concat_smoke/text/text_embeddings_{split}.parquet
.exp/f2_feature_concat_smoke/image/image_embeddings_{split}.parquet
```

Command:

```bash
python -m src.reference_baseline_branch.run_reference_baselines --config .exp/f2_feature_concat_smoke/f2_smoke_config.yaml --baseline F2-LGBM-Concat-Smoke --target popularity
```

Result:

```text
status = ok
n_train = 9583
n_val = 2918
n_test = 3087
n_features = 159
output = .exp/f2_feature_concat_smoke/results/01/baseline_results.csv
```

Conclusion:

```text
The F2 feature-concat architecture is wired correctly for metadata + text parquet + image parquet: ID alignment, feature concatenation, model training, metric reporting, and result writing all run end to end.
```

Remaining blockers for the real `F2-XGB-Concat` run:

```text
resolved
```

## 16. Real Embedding Run

Checked on 2026-05-12:

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline F2-XGB-Concat
```

Output:

```text
.exp/baseline/results/14/baseline_results.csv
```

Result:

| target | n_train | n_val | n_test | n_features | test_MAE | test_R2 | test_Spearman_rho | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| popularity | 9205 | 2637 | 2808 | 1559 | 9588.2590 | 0.5194 | 0.8575 | ok |
| meanScore | 9205 | 2637 | 2808 | 1559 | 8.3391 | 0.0193 | 0.5292 | ok |

Recommended immediate order:

1. Keep `F2-XGB-Concat` as the feature-concat reference baseline result.
2. Decide whether missing text embedding coverage should remain strict-intersection only or become a separately named ablation.
3. Move to `C1-Armenta-MLP` and compare against `F2-XGB-Concat`.
