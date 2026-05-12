# Reference Baseline 狀態紀錄

更新日期：2026-05-12

本文件記錄 `docs/baseline_reference_implementation_plan.md` 目前已實作與已跑通的 reference baseline 範圍。注意：這裡只記錄「文獻/外部比較 baseline」，不包含本專案自己的 ablation baseline。

## 摘要

目前完成的是：

1. `0. Lowest Reference / 最低地板`
2. `1.1 Metadata-only Classical ML`

已經有 first-pass 結果，但尚未完整完成的是：

1. `2.3 Retrieval / RAG Competitive Baseline` 的 `hybrid` / selective retrieval 延伸

換句話說：目前已完成 foundation/classical、single-modality、feature-concat、`C1-Armenta-MLP` first-pass adaptation、`C1-Armenta-ProxyBranchMLP` branch-wise proxy adaptation、lightweight `C2-CTNN-Lite` cross-modal transformer adaptation，以及 C3 的 first-pass `none/sparse/dense` retrieval baselines。`C1-Armenta-ProxyBranchMLP` 比 flat C1 更接近 Armenta 的 branch-fusion 思路，但仍不是框架重現；C3 目前也只能稱 SKAPP-inspired。

論文復現限制已統一記錄於：

```text
reports/reference_baseline_paper_alignment_audit.md
```

該 audit 覆蓋 C1、C2、C3 的復現限制。簡單說：C1 目前缺 character-description / portrait artifacts；C2 是 lightweight cross-modal transformer transfer，不是完整 CTNN；C3/RAG 在沒有 RRCP-style selection 與 graph/attention fusion 前，只能稱為 SKAPP-inspired。

## 規劃對照

| 規劃路線 | Baseline IDs | 狀態 | 備註 |
|---|---|---|---|
| `0. Lowest Reference / 最低地板` | `F0-Mean`, `F0-Ridge-Meta` | 已完成 | Mean predictor 與 Ridge metadata baseline 已跑 `popularity`、`meanScore` |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | 已完成，屬 adaptation | 對應 Lo & Syu 2023 的 pre-broadcast metadata + classical ML 參考；RF 有原文方法支撐，Gradient Boosting 是本專案延伸強 tabular baseline，不是原文模型 |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | 已完成，屬 adaptation | 已補 `docs/reference_baselines/f2_feature_concat_plan.md`；metadata + text embedding + image embedding concat 架構已用真實 embeddings 與 XGBoost 跑通 `popularity`、`meanScore` |
| `1.3 Text-only Baseline` | `T2-XGB-TextEmb` | 已完成，屬 adaptation | 使用 text embeddings + XGBoost；對應 implementation plan 的 `T2-Emb` 類型，不是 TF-IDF exact reproduction |
| `1.4 Image-only Baseline` | `I1-XGB-ImageEmb` | 已完成，屬 adaptation | 使用 image embeddings + XGBoost；對應 implementation plan 的 `I1-Emb` 類型，不是 poster CNN exact reproduction |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP`, `C1-Armenta-ProxyBranchMLP` | branch-wise proxy 已完成，屬 adaptation | flat C1 與 branch-wise proxy 均已跑通；proxy branch model 更接近 Armenta 的 branch-fusion pattern，但缺 main-character description / portrait artifacts，仍不能稱 reproduction |
| `2.2 Cross-modal Transformer Fusion` | `C2-CTNN-Lite` | 已完成，屬 adaptation | 使用 text/image embedding token + lightweight TransformerEncoder；對應 Madongo et al. 2023 的 cross-modal transformer route，但不是 exact CTNN reproduction |
| `2.3 Retrieval / RAG Competitive Baseline` | `C3-RAG-None-XGB`, `C3-RAG-Sparse-XGB`, `C3-RAG-Dense-XGB` | first-pass 已完成，屬 inspired | 使用 offline train-set knowledge base + temporal filtering；目前完成 none/sparse/dense，hybrid artifact test 尚未產生，仍不是 SKAPP reproduction |

## 已實作程式

Reference baseline 程式：

```text
src/reference_baseline_branch/
├── configs/reference_baselines.yaml
├── sklearn_models.py
├── run_reference_baselines.py
└── README.md
```

共用實驗工具：

```text
src/experiment_common/
├── features.py
├── metrics.py
└── README.md
```

Project ablation scaffold，與 reference baselines 分開：

```text
src/ablation_branch/
├── configs/ablation_baselines.yaml
├── run_ablation_baselines.py
└── README.md
```

## 最新 reference run 紀錄

目錄拆分前的 latest full run：

```text
.exp/baseline/results/05/baseline_results.csv
```

重要說明：`.exp/` 已被 ignore，應視為 local experiment output。這份 markdown 才是 tracked status record。

已追蹤的結果摘要：

```text
reports/reference_baseline_results.csv
reports/reference_baseline_runs.md
reports/reference_baseline_paper_alignment_audit.md
```

`reports/reference_baseline_paper_alignment_audit.md` 應視為 claim boundary 的 source of truth，用來判斷某個 baseline 應該描述成 reproduction、adaptation、proxy adaptation，或只是 inspiration-only。

最新 F2 feature-concat run：

```text
.exp/baseline/results/14/baseline_results.csv
```

最新 C1 deep-fusion first-pass run：

```text
.exp/baseline/results/15/baseline_results.csv
```

最新 C1 branch-wise proxy run：

```text
.exp/baseline/results/19/baseline_results.csv
```

最新 single-modality runs：

```text
.exp/baseline/results/16/baseline_results.csv  # T2-XGB-TextEmb
.exp/baseline/results/17/baseline_results.csv  # I1-XGB-ImageEmb
```

最新 C2 cross-modal transformer run：

```text
.exp/baseline/results/18/baseline_results.csv
```

最新 C3 retrieval runs：

```text
.exp/baseline/results/21/baseline_results.csv  # C3-RAG-None-XGB
.exp/baseline/results/22/baseline_results.csv  # C3-RAG-Sparse-XGB
.exp/baseline/results/23/baseline_results.csv  # C3-RAG-Dense-XGB
```

Strict intersection 使用的 embedding coverage：

| split | metadata ids | text ids in split | image ids in split | 使用的 common ids |
|---|---:|---:|---:|---:|
| train | 9583 | 9205 | 9583 | 9205 |
| val | 2918 | 2637 | 2918 | 2637 |
| test | 3087 | 2808 | 3087 | 2808 |

## 論文對齊檢查

### Claim Boundary 表

| baseline_id | reproduction_level | 論文支撐的部分 | 專案 adaptation 的部分 | 可允許寫法 | 不可允許寫法 |
|---|---|---|---|---|---|
| `F0-Mean` | common | 一般最低參考 baseline，未宣稱復現特定論文。 | 對所有 validation/test samples 預測 train-set target mean。 | 可作為 lowest-reference baseline。 | 不可引用為任何論文 reproduction。 |
| `F0-Ridge-Meta` | common | 一般 linear baseline，未宣稱復現特定論文。 | 使用 AniList pre-release metadata 與本專案 temporal split 做 Ridge regression。 | 可作為 simple linear metadata baseline。 | 不可引用為 Lo & Syu 2023 或其他論文 reproduction。 |
| `F1-RF-Meta` | adapted | Lo & Syu 2023 支撐 pre-broadcast entertainment metadata + classical ML，且包含 Random Forest。 | 用 AniList pre-release metadata 與 temporal split 對 `popularity` / `meanScore` 做 anime-domain regression。 | 可寫成 adapted from Lo & Syu 2023 的 metadata-only Random Forest baseline。 | 不可稱 exact reproduction；原論文是 Japanese TV drama high/low rating classification 並使用 cross-validation。 |
| `F1-GB-Meta` | adapted | Lo & Syu 2023 支撐 metadata-only classical ML 路線，但不特別支撐 Gradient Boosting。 | 在相同 AniList metadata feature set 上做更強的 sklearn Gradient Boosting tabular baseline。 | 可寫成 Lo & Syu 2023 metadata-only classical ML route 的 gradient-boosting extension。 | 不可寫成 Lo & Syu 2023 中出現的模型。 |
| `F2-XGB-Concat` | adapted | Chen et al. 2019 與 Jeong et al. 2024 支撐 visual/textual feature fusion + XGBoost-style classical regressors。 | AniList metadata + project text embeddings + project image embeddings concat 後做 XGBoost regression。 | 可寫成 visual-textual popularity prediction literature adapted feature-concat XGBoost baseline。 | 不可宣稱使用相同 dataset、encoders，或 exact feature extraction。 |
| `C1-Armenta-MLP` | adapted | Armenta-Segura & Sidorov 2025 支撐 anime-domain multimodal prediction，包含 GPT-2 text branches、ResNet-50 visual features、MLP fusion。 | Project metadata + precomputed text/image embeddings + sklearn MLP fusion head；缺少原論文 separate character-description 與 character-portrait branches。 | 可寫成受 Armenta-Segura & Sidorov 2025 啟發的 loose anime-domain multimodal MLP adaptation。 | 不可宣稱 exact framework reproduction、numerical comparability 或 matching paper inputs；原論文用 synopsis、character descriptions、character portraits，不是本專案 metadata + cover/banner embeddings。 |

### `F0-Mean` / `F0-Ridge-Meta`

這兩個是一般最低參考 baseline，不復現特定論文。

### `F1-RF-Meta`

這是 Lo & Syu (2023) 的 project adaptation，不是 exact reproduction。

原論文支撐的是：

- Entertainment-domain pre-broadcast metadata 可用於預測。
- 原論文使用 Japanese prime-time TV drama metadata，例如 broadcast year、season、time slot、station、genre、original/sequel status、screenwriter、cast。
- 原論文評估 classical machine learning classifiers，其中包含 Random Forest。
- 原論文是 metadata classification，用於 high/low rating groups，不是 anime regression。

本專案改動的是：

- Domain 從 Japanese TV dramas 改為 anime。
- Target 從 high/low rating classification 改為 `popularity` 與 `meanScore` regression。
- Split policy 從原論文 cross-validation 改為本專案 temporal train/val/test split。
- Metadata fields 是 AniList-equivalent pre-release fields，不是原論文 exact drama metadata fields。

### `F1-GB-Meta`

這不是 Lo & Syu (2023) 直接提出的模型。它是在同一條 `Metadata-only Classical ML` 路線下加入的更強 classical tabular extension。報告時應寫成「Lo & Syu 2023 + gradient boosting adaptation」，不是 direct reproduction。

### 已完成結果快照

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
| `C1-Armenta-MLP` | popularity | 15352.2529 | -0.9811 | 0.8250 | ok |
| `C1-Armenta-MLP` | meanScore | 9.0610 | -0.1173 | 0.4494 | ok |
| `C1-Armenta-ProxyBranchMLP` | popularity | 13663.7819 | 0.2600 | 0.8490 | ok |
| `C1-Armenta-ProxyBranchMLP` | meanScore | 8.2724 | 0.0398 | 0.4763 | ok |
| `T2-XGB-TextEmb` | popularity | 14908.8897 | -0.0152 | 0.6488 | ok |
| `T2-XGB-TextEmb` | meanScore | 10.3206 | -0.3846 | 0.2427 | ok |
| `I1-XGB-ImageEmb` | popularity | 13815.0865 | 0.0158 | 0.6046 | ok |
| `I1-XGB-ImageEmb` | meanScore | 9.4042 | -0.1559 | 0.2918 | ok |
| `C2-CTNN-Lite` | popularity | 13764.4086 | 0.1716 | 0.7410 | ok |
| `C2-CTNN-Lite` | meanScore | 9.5102 | -0.2602 | 0.3107 | ok |
| `C3-RAG-None-XGB` | popularity | 9664.2004 | 0.5064 | 0.8583 | ok |
| `C3-RAG-None-XGB` | meanScore | 8.3647 | 0.0132 | 0.5307 | ok |
| `C3-RAG-Sparse-XGB` | popularity | 9736.1037 | 0.5725 | 0.8722 | ok |
| `C3-RAG-Sparse-XGB` | meanScore | 8.1703 | 0.0730 | 0.5384 | ok |
| `C3-RAG-Dense-XGB` | popularity | 9704.8621 | 0.5084 | 0.8584 | ok |
| `C3-RAG-Dense-XGB` | meanScore | 8.2445 | 0.0464 | 0.5382 | ok |

### F2 架構 smoke test

2026-05-12，先用 synthetic text/image embeddings 測試 feature-concat path：

```text
python -m src.reference_baseline_branch.run_reference_baselines --config .exp/f2_feature_concat_smoke/f2_smoke_config.yaml --baseline F2-LGBM-Concat-Smoke --target popularity
```

結果：

```text
status = ok
n_train = 9583
n_val = 2918
n_test = 3087
n_features = 159
```

這確認 runner 在真實 embeddings 到位前，已能完成 metadata + text embedding parquet + image embedding parquet 的 ID alignment、feature concatenation、model fitting、metrics、result output。這個 smoke result 不能作為正式 baseline 回報，因為 embeddings 是 synthetic，而且 smoke model 使用 LightGBM，不是 XGBoost。

### F2 真實 embedding run

2026-05-12，真實 text/image embeddings 已放到設定路徑下，並在目前 Python environment 安裝 `xgboost`。

指令：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline F2-XGB-Concat
```

結果：

```text
status = ok
n_train = 9205
n_val = 2637
n_test = 2808
n_features = 1559
```

### C1 first-pass deep fusion run

2026-05-12，使用真實 text/image embeddings 跑 `C1-Armenta-MLP`：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-MLP --include-disabled
```

結果：

```text
status = ok
n_train = 9205
n_val = 2637
n_test = 2808
n_features = 1559
```

論文對齊註記：

```text
這是 loose adaptation，不是 framework reproduction。原論文使用 GPT-2 synopsis、GPT-2 main-character descriptions、ResNet-50 main-character portraits 與 MLP fusion；目前實作使用 project metadata 加 precomputed text/image embeddings。
```

目前解讀：

```text
First-pass sklearn MLP fusion head 沒有超過 `F2-XGB-Concat`。報告時應視為 adapted deep-fusion attempt，而不是 deep fusion 優於 feature concatenation 的證據。
```

### C1 branch-wise proxy fusion run

2026-05-12，新增並執行 `C1-Armenta-ProxyBranchMLP`：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-ProxyBranchMLP --include-disabled
```

實作：

```text
metadata proxy -> metadata branch MLP
text embedding -> text branch MLP
image embedding -> image branch MLP
branch outputs -> fusion MLP -> regression head
```

結果：

```text
status = ok
n_train = 9205
n_val = 2637
n_test = 2808
n_features = 1559
```

論文對齊註記：

```text
這比 flat `C1-Armenta-MLP` 更接近 Armenta-Segura & Sidorov 2025，因為它保留 branch-wise modality processing 再做 fusion。不過它仍是 proxy adaptation，因為目前 artifacts 不包含 main-character descriptions 或 main-character portraits。
```

目前解讀：

```text
`C1-Armenta-ProxyBranchMLP` 明顯優於 flat C1 MLP。它在 popularity R2 仍落後 `F2-XGB-Concat`，但 meanScore R2 略有改善，同時犧牲部分 rank correlation。
```

### Single-modality runs

2026-05-12，新增並執行 text-only 與 image-only embedding baselines：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline T2-XGB-TextEmb
python -m src.reference_baseline_branch.run_reference_baselines --baseline I1-XGB-ImageEmb
```

結果：

| baseline_id | n_train | n_val | n_test | n_features |
|---|---:|---:|---:|---:|
| `T2-XGB-TextEmb` | 9205 | 2637 | 2808 | 384 |
| `I1-XGB-ImageEmb` | 9583 | 2918 | 3087 | 1024 |

目前解讀：

```text
Text-only 與 image-only embeddings 都含有 ranking signal，特別是 popularity；但單獨使用時都不足以產生強 R2。`F2` 的增益看起來主要來自 metadata 與 embeddings 的組合，而不是任一 embedding modality 單獨造成。
```

### C2 cross-modal transformer run

2026-05-12，新增並執行 `C2-CTNN-Lite`：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-CTNN-Lite --include-disabled
```

實作：

```text
text embedding -> projection -> text token
image embedding -> projection -> image token
two-token TransformerEncoder -> pooled fusion vector -> regression head
```

論文對齊註記：

```text
這是 lightweight adaptation，不是 CTNN reproduction。原論文使用 poster/review deep features、cross-modal attention transformers、recurrent fusion、metadata-related factors 與 box-office class prediction；目前實作使用 precomputed anime text/image embeddings 與 simplified regression head。
```

結果：

```text
status = ok
n_train = 9205
n_val = 2637
n_test = 2808
n_features = 1408
```

目前解讀：

```text
`C2-CTNN-Lite` 在 popularity 上優於 text-only/image-only embeddings，但仍落後 `F2-XGB-Concat`。對 meanScore 來說，第一版 cross-modal transformer adaptation 仍偏弱。
```

### C3 first-pass retrieval runs

2026-05-12，新增 offline C3 RAG feature builder，並執行三組 first-pass retrieval baselines：

```text
python -m src.reference_baseline_branch.build_c3_rag_features --modes none sparse dense hybrid --top-k 10
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-None-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Sparse-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Dense-XGB --include-disabled
```

實作：

```text
train set -> temporally filtered knowledge base
none   -> schema-compatible no-retrieval RAG features
sparse -> genre/studio/voice_actor/source overlap retrieval
dense  -> text embedding semantic retrieval
top-k retrieved items -> aggregate RAG features -> XGBoost with metadata/text/image
```

結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho |
|---|---:|---:|---:|---:|
| `C3-RAG-None-XGB` | popularity | 9664.2004 | 0.5064 | 0.8583 |
| `C3-RAG-Sparse-XGB` | popularity | 9736.1037 | 0.5725 | 0.8722 |
| `C3-RAG-Dense-XGB` | popularity | 9704.8621 | 0.5084 | 0.8584 |
| `C3-RAG-None-XGB` | meanScore | 8.3647 | 0.0132 | 0.5307 |
| `C3-RAG-Sparse-XGB` | meanScore | 8.1703 | 0.0730 | 0.5384 |
| `C3-RAG-Dense-XGB` | meanScore | 8.2445 | 0.0464 | 0.5382 |

論文對齊註記：

```text
這些是 SKAPP-inspired retrieval baselines，不是 SKAPP reproduction。它們沒有 RRCP selection、VL-GNN contextual learning 或 RRCP-Attention fusion。
```

目前解讀：

```text
Sparse metadata retrieval 是目前 C3 first-pass 最強版本，popularity test_R2 從 no-RAG 的 0.5064 提升到 0.5725，也略高於 F2-XGB-Concat 的 0.5194。Dense semantic retrieval 對 popularity 幾乎接近 no-RAG，但對 meanScore 有小幅改善。Hybrid artifact 目前 train/val 已產生，但 test 在生成時 timeout，尚未列入正式結果。
```

## 復現指令

執行所有 enabled reference baselines：

```bash
python -m src.reference_baseline_branch.run_reference_baselines
```

執行單一 baseline：

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline F1-RF-Meta --target popularity
```

另外檢查 ablation scaffold：

```bash
python -m src.ablation_branch.run_ablation_baselines
```

## 下一步

若要從已完成的 foundation/classical、single-modality、neural-fusion baselines 往下一條 reference route 推進：

1. 若目標是補齊剩下的 anchor paper route，下一步應繼續 `2.3 Retrieval / RAG Competitive Baseline`。
2. 只有在目標變成提升性能，而不是補完整 reference map 時，才優先 tune 或替換 C1/C2 neural fusion heads。
3. 決定目前報告是否讓 `F2-XGB-Concat` 維持 primary competitive reference floor。

只有完成第 1 點後，SKAPP-inspired retrieval anchor paper 才能被正式放進 baseline table。

如果目標從「cover the reference route」改成「closer reproduction」，優先順序應改為：

1. `C1-Armenta-Figure2Proxy`：抽取 raw AniList main-character descriptions 與 portrait URLs，產生 separate character text/image embeddings，然後實作接近 Figure 2 的 character MLP + Big MLP。
2. `C3-SKAPP-Inspired`：把目前 `none/sparse/dense/hybrid` retrieval 升級成 top-k aggregation + contribution filtering，再考慮比 vanilla retrieval 更強的 claim。
3. `C2-CTNN-Proxy`：只有在需要更強 transformer proxy 時，才加入 explicit cross-attention 與 metadata/recurrent-style fusion；exact CTNN reproduction 不太適合目前 anime dataset，因為原論文的 review/poster/movie box-office task 差異太大。
