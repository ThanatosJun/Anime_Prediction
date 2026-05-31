# Baseline Directory Planning

本文件規劃 baseline 相關程式與輸出的邊界。重點是把「文獻參考 baseline」和「本專案自己的消融/對照 baseline」分開，避免後續報告時把 external comparison、internal ablation、main model 混在一起。

## 1. Baseline 分層

### 1.1 Reference Baselines

Reference baselines 是外部比較座標，用來回答：

> 我們的方法和既有文獻支持的合理方法相比如何？

這一層對應 `docs/baseline_reference_implementation_plan.md`，包含：

| 類型 | 例子 | 主要用途 |
|---|---|---|
| lowest reference | Mean, Ridge | 確認任務不是連最低地板都贏不了 |
| metadata classical ML | RF, Gradient Boosting, XGBoost/LightGBM | 對齊 pre-release metadata + classical ML 文獻 |
| feature-concat classical ML | metadata + text_emb + image_emb -> XGBoost/LightGBM | 對齊 visual-textual feature fusion 文獻 |
| domain deep fusion | metadata + text_emb + image_emb -> MLP | 對齊 anime multimodal deep baseline |
| cross-modal transformer | text_emb + image_emb -> attention/transformer fusion | 對齊 poster + text transformer baseline |
| retrieval baseline | no/sparse/dense/hybrid/selective retrieval | 對齊 SKAPP-inspired retrieval baseline |

### 1.2 Project / Ablation Baselines

Project baselines 是本專案內部對照，用來回答：

> 我們自己的模態、模組、資料處理、RAG 設計各自貢獻多少？

這一層不一定要主張「復現某篇論文」，但必須用同一套 split、target transform、metrics 和 leakage rule。

| 類型 | 例子 | 主要用途 |
|---|---|---|
| modality ablation | metadata only, text only, image only, metadata+text, metadata+image, text+image, all modalities | 回答各模態貢獻 |
| image ablation | cover only, banner only, cover+banner, missing image zero/impute | 回答影像特徵是否有效 |
| text ablation | title only, description only, title+description, raw vs supplemented description | 回答文字來源是否有效 |
| RAG ablation | none, sparse, dense, hybrid, time-filter on/off, top-k variants | 回答 retrieval 設計是否有效 |
| fusion ablation | concat MLP, projected MLP, gated fusion, attention fusion | 回答融合架構是否有效 |
| target/data ablation | full data vs post-2000, popularity raw/log, meanScore shift diagnostics | 回答資料切分與 target 處理影響 |

## 2. 建議目錄結構

```text
src/
├── reference_baseline_branch/      # 外部文獻 baseline 實作
│   ├── configs/
│   │   └── reference_baselines.yaml
│   ├── sklearn_models.py          # Mean/Ridge/RF/GB/XGB/LightGBM factories
│   ├── armenta_fusion.py          # Anime domain MLP fusion adaptation, planned
│   ├── ctnn.py                    # cross-modal transformer baseline, optional
│   ├── retrieval.py               # SKAPP-inspired retrieval baselines, optional
│   └── run_reference_baselines.py
│
├── ablation_branch/                # 本專案內部 baseline / ablation
│   ├── configs/
│   │   └── ablation_baselines.yaml
│   ├── modality.py                # metadata/text/image combinations, planned
│   ├── image.py                   # cover/banner image ablations, planned
│   ├── text.py                    # text source ablations, planned
│   ├── rag.py                     # none/sparse/dense/hybrid/top-k/time filter, planned
│   ├── fusion.py                  # concat/projection/gating/attention variants, planned
│   └── run_ablation_baselines.py
│
└── experiment_common/
    ├── features.py                # shared feature loading and train-only encoders
    ├── metrics.py                 # shared metrics
    ├── registry.py                # baseline id -> runner mapping, planned
    └── reporting.py               # result table and markdown summary, planned
```

目前已實作的第一版已整理為 `reference_baseline_branch` 的 vertical slice：

```text
src/experiment_common/features.py
src/experiment_common/metrics.py
src/reference_baseline_branch/sklearn_models.py
src/reference_baseline_branch/run_reference_baselines.py
src/reference_baseline_branch/configs/reference_baselines.yaml
src/ablation_branch/configs/ablation_baselines.yaml
```

## 3. Baseline ID 命名規則

建議 ID 直接反映 baseline 層級。

| Prefix | 意義 | 例子 |
|---|---|---|
| `F0` | lowest reference | `F0-Mean`, `F0-Ridge-Meta` |
| `F1` | foundation metadata/classical | `F1-RF-Meta`, `F1-GB-Meta` |
| `F2` | foundation feature concat | `F2-XGB-Concat-All` |
| `T` | text-only / text ablation | `T1-TFIDF-Desc`, `T2-Emb-TitleDesc` |
| `I` | image-only / image ablation | `I1-Cover`, `I2-CoverBanner` |
| `A` | project ablation | `A1-Modality-All`, `A2-NoImage`, `A3-NoRAG` |
| `C` | competitive reference baseline | `C1-Armenta-MLP`, `C2-CTNN`, `C3-SKAPP-Inspired` |
| `M` | main project model candidate | `M1-FusionMLP-HybridRAG` |

重要：`A*` 和 `C*` 要分開。`A*` 是我們自己的消融，`C*` 是文獻支撐的 competitive baseline。

## 4. 統一輸出格式

所有 reference baseline 和 ablation baseline 都應輸出同一個 schema，方便最後合併。

```text
baseline_id
baseline_group        # reference / ablation / main
target
feature_set
model
reference             # ablation 可填 "project ablation"
split_policy
data_scope            # full / post2000 / other
uses_metadata
uses_text
uses_image
uses_rag
val_MAE
val_RMSE
val_R2
val_Spearman_rho
val_log_MAE
test_MAE
test_RMSE
test_R2
test_Spearman_rho
test_log_MAE
status
notes
```

## 5. 報告解讀規則

Reference baselines 放在論文/報告的「和既有方法比較」段落。

Project ablations 放在 RQ 或 analysis 段落，例如：

- RQ1: metadata/text/image 各模態是否有效？
- RQ2: retrieval 是否改善 pre-release prediction？
- RQ3: fusion architecture 是否比簡單 concat 更有效？

因此結果表建議拆成三張：

1. `reference_baseline_results.csv`
2. `ablation_results.csv`
3. `main_model_results.csv`

最後再產生一張整合表：

4. `all_experiment_results.csv`

## 6. 下一步

短期建議：

1. 先完成 reference foundation baselines。
2. 產生 text/image embeddings 後，補跑 `F2-XGB-Concat` 和 `C1-Armenta-MLP`。
3. 實作 `src/ablation_branch/run_ablation_baselines.py`，把 modality/text/image/RAG/fusion 消融結果輸出成同一個 schema。
4. 把 `src/fussion_branch/run_rq2_rag_experiments.py` 的 RAG 對照結果整理進 ablation schema。
