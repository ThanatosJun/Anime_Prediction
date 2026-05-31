# Reference Baseline 狀態紀錄

更新日期：2026-05-19

本文件記錄 `docs/baseline_reference_implementation_plan.md` 目前已實作與已跑通的 reference baseline 範圍。注意：這裡只記錄「文獻/外部比較 baseline」，不包含本專案自己的 ablation baseline。

## 摘要

目前已完成 foundation/classical、single-modality、feature-concat、C1 first-pass/proxy/project-input reconstruction、C1 Figure 2 character-centric side reconstruction、C2 first-pass/proxy/project-input CTNN reconstruction、C3 first-pass retrieval/proxy/graph/full reconstruction runs。

目前主線判準只有兩條：

1. Baseline 輸入必須對齊本專案主框架，也就是 metadata、synopsis/text embedding、cover/banner image，以及必要時的 project retrieval context。
2. 在輸入固定為本專案主框架的前提下，模型方法才往原論文設計盡量還原。

`C1-Armenta-ProjectInputReconstruction` 是目前 C1 主線：保留 project metadata/synopsis/cover-banner contract，補上 GPT-2 synopsis、ResNet-50 cover/banner features、project-context MLP 與 Armenta Big MLP。`C1-Armenta-Figure2Reconstruction` 也已完成，但它改用 main-character description/name 與 portrait artifacts，所以只能作 Figure 2 side analysis，不能取代主線 project-input row。

論文復現限制已統一記錄於：

```text
reports/reference_baseline_paper_alignment_audit.md
```

該 audit 覆蓋 C1、C2、C3 的復現限制。簡單說：C1 主線採 project-input reconstruction，Figure 2 character route 只作旁支；C2 主線採 project-input CTNN reconstruction，dual-visual row 只作 ResNet50+ViT-like 對齊診斷；C3 主線保留 project historical-anime retrieval，再補 learned contribution scoring 與 graph/attention fusion。任何換掉本專案主輸入契約的版本都不能列為主線 baseline。

## 規劃對照

| 規劃路線 | Baseline IDs | 狀態 | 備註 |
|---|---|---|---|
| `0. Lowest Reference / 最低地板` | `F0-Mean`, `F0-Ridge-Meta` | 已完成 | Mean predictor 與 Ridge metadata baseline 已跑 `popularity`、`meanScore` |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | 已完成，屬 adaptation | 對應 Lo & Syu 2023 的 pre-broadcast metadata + classical ML 參考；RF 有原文方法支撐，Gradient Boosting 是本專案延伸強 tabular baseline，不是原文模型 |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | 已完成，屬 adaptation | 已補 `docs/reference_baselines/f2_feature_concat_plan.md`；metadata + text embedding + image embedding concat 架構已用真實 embeddings 與 XGBoost 跑通 `popularity`、`meanScore` |
| `1.3 Text-only Baseline` | `T2-XGB-TextEmb` | 已完成，屬 adaptation | 使用 text embeddings + XGBoost；對應 implementation plan 的 `T2-Emb` 類型，不是 TF-IDF exact reproduction |
| `1.4 Image-only Baseline` | `I1-XGB-ImageEmb` | 已完成，屬 adaptation | 使用 image embeddings + XGBoost；對應 implementation plan 的 `I1-Emb` 類型，不是 poster CNN exact reproduction |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP`, `C1-Armenta-ProxyBranchMLP`, `C1-Armenta-ProjectInputProxy`, `C1-Armenta-ProjectInputProxy-ResNet50`, `C1-Armenta-ProjectInputReconstruction`, `C1-Armenta-Figure2Reconstruction` | project-input reconstruction 與 Figure 2 side reconstruction 已完成 | 主線採 `C1-Armenta-ProjectInputReconstruction`；Figure 2 row 更貼近原論文 character branch，但因輸入不對齊本專案 cover/banner 主框架，只列旁支分析 |
| `2.2 Cross-modal Transformer Fusion` | `C2-CTNN-Lite`, `C2-ProjectInputCrossAttention`, `C2-ProjectInputRecurrentFusion`, `C2-ProjectInputCTNNReconstruction`, `C2-ProjectInputCTNNDualVisualReconstruction` | project-input reconstruction 與 dual-visual diagnostic 已完成 | CTNN 主線採 `C2-ProjectInputCTNNReconstruction`；dual-visual row 補上 ResNet50 + ViT-like 雙視覺來源，但目前只作 source-alignment diagnostic |
| `2.3 Retrieval / RAG Competitive Baseline` | `C3-RAG-None-XGB`, `C3-RAG-Sparse-XGB`, `C3-RAG-Dense-XGB`, `C3-RAG-Hybrid-XGB`, `C3-RAG-Selective-XGB`, `C3-ProjectInputSKAPPProxy-XGB` | project-aligned RAG 主線已完成，屬 inspired | 使用 offline train-set knowledge base + temporal filtering；`C3-RAG-Selective-XGB` 是目前 RAG route strongest row；`C3-ProjectInputSKAPPProxy-XGB` 則是目前較接近 SKAPP selective retrieval 動機的對齊版 proxy，但仍不是 SKAPP reproduction |

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

最新 C1 project-input proxy run：

```text
.exp/baseline/results/26/baseline_results.csv
```

最新 C1 project-input ResNet-50 visual-encoder proxy run：

```text
.exp/baseline/results/30/baseline_results.csv
```

最新 C1 project-input reconstruction run：

```text
.exp/baseline/results/36/baseline_results.csv
```

最新 C1 Figure 2 side reconstruction run：

```text
.exp/baseline/results/38/baseline_results.csv
```

最新 C2 project-input CTNN reconstruction run：

```text
.exp/baseline/results/37/baseline_results.csv
```

最新 C2 dual-visual CTNN diagnostic run：

```text
.exp/baseline/results/39/baseline_results.csv
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
.exp/baseline/results/24/baseline_results.csv  # C3-RAG-Hybrid-XGB
.exp/baseline/results/25/baseline_results.csv  # C3-RAG-Selective-XGB
.exp/baseline/results/33/baseline_results.csv  # C3-ProjectInputSKAPPProxy-XGB
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
| `C1-Armenta-ProjectInputProxy` | popularity | 11951.3799 | 0.3819 | 0.8366 | ok |
| `C1-Armenta-ProjectInputProxy` | meanScore | 8.7567 | -0.0678 | 0.4591 | ok |
| `C1-Armenta-ProjectInputProxy-ResNet50` | popularity | 11482.0224 | 0.3817 | 0.8152 | ok |
| `C1-Armenta-ProjectInputProxy-ResNet50` | meanScore | 8.7347 | -0.0633 | 0.4586 | ok |
| `C1-Armenta-ProjectInputReconstruction` | popularity | 10719.7513 | 0.2898 | 0.8192 | ok |
| `C1-Armenta-ProjectInputReconstruction` | meanScore | 9.0250 | -0.1096 | 0.4666 | ok |
| `C1-Armenta-Figure2Reconstruction` | popularity | 11878.6328 | 0.3556 | 0.7823 | ok |
| `C1-Armenta-Figure2Reconstruction` | meanScore | 9.7747 | -0.2172 | 0.3824 | ok |
| `T2-XGB-TextEmb` | popularity | 14908.8897 | -0.0152 | 0.6488 | ok |
| `T2-XGB-TextEmb` | meanScore | 10.3206 | -0.3846 | 0.2427 | ok |
| `I1-XGB-ImageEmb` | popularity | 13815.0865 | 0.0158 | 0.6046 | ok |
| `I1-XGB-ImageEmb` | meanScore | 9.4042 | -0.1559 | 0.2918 | ok |
| `C2-CTNN-Lite` | popularity | 13764.4086 | 0.1716 | 0.7410 | ok |
| `C2-CTNN-Lite` | meanScore | 9.5102 | -0.2602 | 0.3107 | ok |
| `C2-ProjectInputCrossAttention` | popularity | 12755.1921 | 0.3704 | 0.8379 | ok |
| `C2-ProjectInputCrossAttention` | meanScore | 8.0600 | 0.0597 | 0.4663 | ok |
| `C2-ProjectInputRecurrentFusion` | popularity | 12493.3574 | 0.3545 | 0.8498 | ok |
| `C2-ProjectInputRecurrentFusion` | meanScore | 8.0768 | 0.0670 | 0.4723 | ok |
| `C2-ProjectInputCTNNReconstruction` | popularity | 10151.2161 | 0.4608 | 0.8471 | ok |
| `C2-ProjectInputCTNNReconstruction` | meanScore | 8.1751 | 0.0696 | 0.5247 | ok |
| `C2-ProjectInputCTNNDualVisualReconstruction` | popularity | 10214.6356 | 0.4421 | 0.8491 | ok |
| `C2-ProjectInputCTNNDualVisualReconstruction` | meanScore | 8.8957 | -0.0720 | 0.5310 | ok |
| `C3-RAG-None-XGB` | popularity | 9664.2004 | 0.5064 | 0.8583 | ok |
| `C3-RAG-None-XGB` | meanScore | 8.3647 | 0.0132 | 0.5307 | ok |
| `C3-RAG-Sparse-XGB` | popularity | 9736.1037 | 0.5725 | 0.8722 | ok |
| `C3-RAG-Sparse-XGB` | meanScore | 8.1703 | 0.0730 | 0.5384 | ok |
| `C3-RAG-Dense-XGB` | popularity | 9704.8621 | 0.5084 | 0.8584 | ok |
| `C3-RAG-Dense-XGB` | meanScore | 8.2445 | 0.0464 | 0.5382 | ok |
| `C3-RAG-Hybrid-XGB` | popularity | 10327.0456 | 0.4828 | 0.8537 | ok |
| `C3-RAG-Hybrid-XGB` | meanScore | 8.3798 | 0.0307 | 0.5539 | ok |
| `C3-RAG-Selective-XGB` | popularity | 9782.2338 | 0.5775 | 0.8746 | ok |
| `C3-RAG-Selective-XGB` | meanScore | 8.0914 | 0.0905 | 0.5470 | ok |
| `C3-ProjectInputSKAPPProxy-XGB` | popularity | 10239.2909 | 0.5170 | 0.8574 | ok |
| `C3-ProjectInputSKAPPProxy-XGB` | meanScore | 8.1715 | 0.0744 | 0.5369 | ok |

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

### C1 project-input Armenta-shaped proxy run

2026-05-13，新增並執行 `C1-Armenta-ProjectInputProxy`：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-ProjectInputProxy --include-disabled
```

實作：

```text
project synopsis/text embedding -> synopsis branch -> 768-dim synopsis vector
project metadata + cover/banner image embedding -> project-context MLP -> 768-dim context vector
synopsis vector + context vector -> Big MLP -> regression head
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
這是 project-input Armenta-shaped proxy，不是 Figure 2 reproduction。它仍使用 project embeddings，尚未重新抽 GPT-2 / ResNet-50 encoder outputs，也沒有 character-description / portrait branches；但它比 C1-Armenta-ProxyBranchMLP 更接近原論文的 synopsis branch + context/character MLP + Big MLP 融合形狀。
```

目前解讀：

```text
`C1-Armenta-ProjectInputProxy` 在 popularity 上明顯優於舊的 `C1-Armenta-ProxyBranchMLP`，test_R2 從 0.2600 提升到 0.3819，但仍低於 `F2-XGB-Concat` 與目前最強的 C3 selective retrieval。meanScore 則退步到 -0.0678，表示 Armenta-shaped Big MLP 對 project inputs 不是全面改善。
```

### C1 project-input ResNet-50 visual-encoder proxy run

2026-05-14，新增並執行 `C1-Armenta-ProjectInputProxy-ResNet50`：

```text
$env:TORCH_HOME='.cache/torch'
python -m src.reference_baseline_branch.build_resnet50_image_embeddings --splits train val test --batch-size 128 --device cuda
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-ProjectInputProxy-ResNet50 --include-disabled
```

ResNet-50 feature artifacts：

```text
.exp/baseline/image_features/resnet50/resnet50_image_embeddings_train.parquet  # shape=(9583, 4099)
.exp/baseline/image_features/resnet50/resnet50_image_embeddings_val.parquet    # shape=(2918, 4099)
.exp/baseline/image_features/resnet50/resnet50_image_embeddings_test.parquet   # shape=(3087, 4099)
```

實作：

```text
project cover image -> ImageNet ResNet-50 avg-pool 2048-dim
project banner image -> ImageNet ResNet-50 avg-pool 2048-dim
cover/banner availability masks -> 2 dims
project synopsis/text embedding -> synopsis branch -> 768-dim synopsis vector
project metadata + ResNet-50 cover/banner features -> project-context MLP -> 768-dim context vector
synopsis vector + context vector -> Big MLP -> regression head
```

結果：

| target | n_train | n_val | n_test | n_features | test_MAE | test_R2 | test_Spearman_rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| `popularity` | 9205 | 2637 | 2808 | 4633 | 11482.0224 | 0.3817 | 0.8152 |
| `meanScore` | 9205 | 2637 | 2808 | 4633 | 8.7347 | -0.0633 | 0.4586 |

論文對齊註記：

```text
這是目前 C1 中 visual encoder 最接近 Armenta-Segura & Sidorov 2025 的 project-input proxy，因為它用 ImageNet ResNet-50 抽 cover/banner raw images，而不是直接沿用專案既有 image embeddings。不過它仍不是 Figure 2 reproduction：輸入仍是本專案 cover/banner，不是 main-character portraits；文字也仍是 project text embeddings，不是 GPT-2 synopsis/character-description branches。
```

目前解讀：

```text
ResNet-50 版讓 C1 的 image encoder 更接近原論文，但 popularity test_R2 幾乎等同 `C1-Armenta-ProjectInputProxy`（0.3817 vs 0.3819），Spearman 反而較低（0.8152 vs 0.8366）。meanScore 有極小改善（-0.0633 vs -0.0678），但仍不是強 row。因此它有論文對齊價值，性能上不能取代 F2 或 C3。
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

### C2 cross-modal transformer runs

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

C2 目前主線判斷：

```text
`C2-CTNN-Lite` 可保留為 project-input CTNN-style first pass，但不能作為完整 C2 對齊終點。照兩條主線判準，C2 應先保留 project synopsis/text embedding、cover/banner image embedding 與 metadata，再加入 explicit text-image cross-attention 與 metadata-conditioned fusion。
```

2026-05-14，新增並執行 `C2-ProjectInputCrossAttention`：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputCrossAttention --include-disabled
```

實作：

```text
metadata -> metadata token
text embedding -> text token
image embedding -> image token
text token <-> image token explicit bidirectional cross-attention
metadata token -> modality gates over text/image/metadata tokens
gated fusion vector -> regression head
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
這是 project-input CTNN-style cross-attention proxy，不是 CTNN reproduction。它保留本專案 metadata/text/image inputs，並比 CTNN-Lite 更接近原文 cross-modal attention 與 metadata factors 的動機；但仍使用 anime embeddings/regression target，不是 movie poster/review + box-office class setup。
```

目前解讀：

```text
`C2-ProjectInputCrossAttention` 明顯優於 `C2-CTNN-Lite`：popularity test_R2 從 0.1716 提升到 0.3704，meanScore test_R2 從 -0.2602 提升到 0.0597。不過它仍低於 `F2-XGB-Concat` 的 popularity 0.5194，也低於目前最強的 C3 selective retrieval 0.5775。
```

2026-05-14，補上並執行 `C2-ProjectInputRecurrentFusion`：

```text
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputRecurrentFusion --include-disabled
```

實作：

```text
metadata -> metadata token
text embedding -> text token
image embedding -> image token
text token <-> image token explicit bidirectional cross-attention
[text, image, metadata] token sequence -> GRU recurrent fusion
metadata-conditioned gate over recurrent context
recurrent summary + gated context -> regression head
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
這是 project-input CTNN-style recurrent-fusion proxy，不是 CTNN reproduction。它補上原文 recurrent fusion 的模型動機，但仍保留本專案 metadata/text/image inputs、anime embeddings 與 regression targets。
```

目前解讀：

```text
`C2-ProjectInputRecurrentFusion` 補齊 recurrent-fusion proxy 後，meanScore test_R2 從 CrossAttention 的 0.0597 小幅提升到 0.0670，popularity Spearman 從 0.8379 提升到 0.8498；但 popularity test_R2 從 0.3704 降到 0.3545。因此 recurrent fusion 有對齊價值，但不是整體性能明顯勝出的 C2 版本。
```

### C3 first-pass retrieval runs

2026-05-12 至 2026-05-13，新增 offline C3 RAG feature builder，並執行五組 first-pass retrieval baselines：

```text
python -m src.reference_baseline_branch.build_c3_rag_features --modes none sparse dense hybrid selective --top-k 10
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-None-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Sparse-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Dense-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Hybrid-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Selective-XGB --include-disabled
```

實作：

```text
train set -> temporally filtered knowledge base
none   -> schema-compatible no-retrieval RAG features
sparse -> genre/studio/voice_actor/source overlap retrieval
dense  -> text embedding semantic retrieval
hybrid -> sparse + dense reciprocal-rank fusion
selective -> sparse top-k candidates with median-threshold contribution filtering
top-k retrieved items -> aggregate RAG features -> XGBoost with metadata/text/image
```

結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho |
|---|---:|---:|---:|---:|
| `C3-RAG-None-XGB` | popularity | 9664.2004 | 0.5064 | 0.8583 |
| `C3-RAG-Sparse-XGB` | popularity | 9736.1037 | 0.5725 | 0.8722 |
| `C3-RAG-Dense-XGB` | popularity | 9704.8621 | 0.5084 | 0.8584 |
| `C3-RAG-Hybrid-XGB` | popularity | 10327.0456 | 0.4828 | 0.8537 |
| `C3-RAG-Selective-XGB` | popularity | 9782.2338 | 0.5775 | 0.8746 |
| `C3-RAG-None-XGB` | meanScore | 8.3647 | 0.0132 | 0.5307 |
| `C3-RAG-Sparse-XGB` | meanScore | 8.1703 | 0.0730 | 0.5384 |
| `C3-RAG-Dense-XGB` | meanScore | 8.2445 | 0.0464 | 0.5382 |
| `C3-RAG-Hybrid-XGB` | meanScore | 8.3798 | 0.0307 | 0.5539 |
| `C3-RAG-Selective-XGB` | meanScore | 8.0914 | 0.0905 | 0.5470 |

論文對齊註記：

```text
這些是 SKAPP-inspired retrieval baselines，不是 SKAPP reproduction。`C3-RAG-Selective-XGB` 只使用 deterministic median-threshold contribution proxy，仍沒有 SKAPP 的 RRCP selection、VL-GNN contextual learning 或 RRCP-Attention fusion。
```

目前解讀：

```text
Selective sparse retrieval 是目前 C3 first-pass 最強版本，popularity test_R2 從 no-RAG 的 0.5064 提升到 0.5775，也高於 F2-XGB-Concat 的 0.5194。Plain sparse retrieval 已有穩定增益，selective filtering 又小幅改善；dense semantic retrieval 對 popularity 幾乎接近 no-RAG，但對 meanScore 有小幅改善。Hybrid RRF 沒有超過 sparse/selective；在 popularity R2 上甚至低於 no-RAG，因此目前不支援「多訊號 retrieval 必然更好」的 claim。
```

C3 目前主線判斷：

```text
`C3-RAG-Selective-XGB` 可作為目前 RAG route 的 strongest project-aligned reference row。照兩條主線判準，C3 應先保留 project query anime + historical anime retrieval，再加入 learned contribution scoring/filtering 與 retrieved-set graph/attention fusion。合理下一步是 `C3-ProjectInputSKAPPProxy`，不是把任務換成 social-media UGC popularity。
```

更新：`C3-ProjectInputSKAPPProxy-XGB` 與 `C3-ProjectInputSKAPPGraphProxy`
皆已完成第一版，但只能視為 development milestones。若要讓 reference
comparison 有價值，下一個正式目標必須是 `C3-ProjectInputSKAPPFull`：
完整重做 SKAPP 的 all-items model、dissembled/single-item model、
RRCP_silver、threshold variable-length filtering、GraphLearner、RRCP/CXMI
attention head。`C3-RAG-Selective-XGB` 仍可保留為 ablation/performance row，
但不應取代 structure-complete reconstruction。

### C3 project-input SKAPP-style proxy run

2026-05-14，新增並執行 `C3-ProjectInputSKAPPProxy-XGB`：

```text
python -m src.reference_baseline_branch.build_c3_rag_features --modes skapp_proxy --top-k 10
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-ProjectInputSKAPPProxy-XGB --include-disabled
```

實作：

```text
train set -> temporally filtered historical anime knowledge base
sparse + dense hybrid retrieval -> top-k candidates
train-only query-candidate pairs -> GradientBoosting contribution scorer
learned contribution score -> median-threshold filtering
selected candidates -> softmax attention-weighted aggregate context
metadata + text embedding + image embedding + SKAPP-style RAG features -> XGBoost regression
```

Contribution scorer 訓練：

```text
381552 train-only query-candidate pairs
pair features: sparse/dense/RRF scores, genre/studio/voice overlap, release gap, format/source/country match, episode difference, candidate popularity/score
pair label: train-only popularity closeness + meanScore closeness proxy
```

正式 artifacts：

```text
.exp/baseline/rag_features/skapp_proxy/rag_features_train.parquet
.exp/baseline/rag_features/skapp_proxy/rag_features_val.parquet
.exp/baseline/rag_features/skapp_proxy/rag_features_test.parquet
.exp/baseline/rag_features/skapp_graph_proxy/rag_features_train.parquet
.exp/baseline/rag_features/skapp_graph_proxy/rag_features_val.parquet
.exp/baseline/rag_features/skapp_graph_proxy/rag_features_test.parquet
```

Schema 與 filtering 檢查：

```text
shape = (9583, 22)
skapp_selected_count mean = 4.9834
skapp_selected_count median = 5.0000
skapp_attention_entropy mean = 1.6021

skapp_graph_proxy shape = (9583, 14142)
skapp_graph_proxy feature-store train x = (9205, 15695)
skapp_graph_mask mean = 0.4982

skapp_full tensor dataset train shape = 9583 rows
skapp_full n_features = 15508
RRCP_silver popularity train mean = 0.01533
RRCP_silver meanScore train mean = -0.00236
```

結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|---:|
| `C3-ProjectInputSKAPPProxy-XGB` | popularity | 10239.2909 | 0.5170 | 0.8574 | 0.9704 |
| `C3-ProjectInputSKAPPProxy-XGB` | meanScore | 8.1715 | 0.0744 | 0.5369 |  |
| `C3-ProjectInputSKAPPGraphProxy` | popularity | 11501.8681 | 0.4404 | 0.8561 | 1.0245 |
| `C3-ProjectInputSKAPPGraphProxy` | meanScore | 8.1448 | 0.0690 | 0.4973 |  |
| `C3-ProjectInputSKAPPFull` | popularity | 14668.1228 | -0.4927 | 0.6985 | 1.2983 |
| `C3-ProjectInputSKAPPFull` | meanScore | 9.8063 | -0.2385 | 0.3657 |  |

論文對齊註記：

```text
這是 aggregate 層級最接近 SKAPP selective retrieval 動機的 project-input proxy，因為它補上 learned contribution scoring、learned filtering、attention-weighted retrieved context aggregation。更接近原始碼架構的版本是 `C3-ProjectInputSKAPPGraphProxy`；兩者仍不是 SKAPP reproduction，因為 RRCP/VL-GNN/RRCP-Attention 都仍為 proxy，也不是 social-media UGC task。
```

2026-05-19 更新：`C3-ProjectInputSKAPPFull` 已完成第一版並跑出 run 35。
它已經包含 SKAPP-style tensor dataset、all-items model、single/dissembled
model、RRCP_silver、threshold final RRCP model、GraphLearner-style fusion、
RRCP/CXMI-style weighting。第一版 performance 很差，因此目前代表「完整構造
已可執行」，不是可報告的最佳性能 row。

2026-05-20 診斷：負 R2 不是因為 tensor 檔案壞掉或 target 欄位讀錯。
完整分析見 `reports/c3_skappfull_negative_r2_diagnosis_2026-05-20.md`。
目前判斷如下：

```text
popularity:
  log-space R2 仍為正（val 0.5520 / test 0.4727），但 expm1 回原始尺度後
  出現少數極端高估值，導致 raw-space RMSE/R2 被拖垮。

meanScore:
  test split target mean = 65.4302，但 prediction mean = 59.4302，
  幾乎貼著 train mean = 59.0320，屬 underfit + split shift。

RRCP_silver:
  分布集中在 0 附近，threshold=0 會讓約半數 retrieval item 被選入，
  目前 selection/weighting 訊號偏弱，仍需重新診斷 all-items 與 single-item model。
```

同日已做 post-hoc 穩定化試算：

| target | method | test_R2 | test_MAE | 說明 |
|---|---|---:|---:|---|
| popularity | raw | -0.4927 | 14668.1228 | run 35 原始結果 |
| popularity | clip to train log range | 0.0601 | 13920.3988 | 只處理 expm1 outlier |
| popularity | val log-linear calibration + clip | 0.1295 | 12765.8056 | test 有改善，但仍弱 |
| meanScore | raw | -0.2385 | 9.8063 | run 35 原始結果 |
| meanScore | val linear calibration | 0.0098 | 8.6655 | 修回接近 0，但不夠強 |

這表示負號可以透過輸出校正拉回，但 `C3-ProjectInputSKAPPFull`
仍不能當主性能結果；下一步要看 all-items/single-item/RRCP_silver 哪一層訊號不足。

SKAPP 原始碼核對：

```text
baseline_refer/skapp-main/src/dataset.py
baseline_refer/skapp-main/src/RRCP/RRCP.py
baseline_refer/skapp-main/src/RRCP_prediction_variable_lenth.py
baseline_refer/skapp-main/src/graph_attention.py
baseline_refer/skapp-main/src/graph_variable_length.py
```

原始碼顯示 SKAPP final model 不是吃 aggregate RAG 欄位，而是吃：

```text
mean_pooling_vec
merged_text_vec
retrieved_visual_feature_embedding_cls
retrieved_textual_feature_embedding
retrieved_label_list
RRCP_silver
label
```

`RRCP_silver` 由 all-items model 和 single-item / dissembled model 產生，用來估計每個 retrieved item 對 prediction 的相對貢獻；final `RRCP_prediction` 會用 `RRCP > threshold` 過濾 retrieved items，再把 query + selected retrieved visual/text features 送進 `GraphLearner`，最後用 RRCP/CXMI 權重和 retrieved labels 做 prediction。

目前解讀：

```text
`C3-ProjectInputSKAPPProxy-XGB` 的方法對齊比 `C3-RAG-Selective-XGB` 更完整，但 performance 沒有更強：popularity test_R2 0.5170 低於 selective sparse 的 0.5775；meanScore test_R2 0.0744 也低於 selective sparse 的 0.0905。因此報告時應把 `C3-RAG-Selective-XGB` 當 strongest C3 row，把 `C3-ProjectInputSKAPPProxy-XGB` 當最接近 SKAPP motivation 的對齊版 proxy。
```

原始碼核對後的下一層：

```text
要讓 C3 更接近 SKAPP，下一步不是繼續增加 aggregate RAG columns，而是做 C3-ProjectInputSKAPPGraphProxy；目前已完成第一版：
1. 產生 retrieved-set tensor artifacts：retrieved text/image、retrieved labels、RRCP-like contribution；query text/image 則沿用 project input embeddings。
2. 用 RRCP threshold mask 保留 selected retrieved items。
3. 實作 GraphLearner-style text/text and image/text graph fusion。
4. 用 RRCP-weighted context + retrieved label embedding 做 prediction head。
```

注意：SKAPP 原始碼的 retrieval pool 會合併 train + valid；本專案的 GraphProxy 仍維持 temporal / train-only retrieval restriction，避免 pre-release evaluation leakage。

C3 差距決策表：

| 項目 | 決策 | 說明 |
|---|---|---|
| Train-only temporal retrieval | 保留 | 不照搬 SKAPP train+valid retrieval pool，避免 pre-release evaluation leakage。 |
| `retrieval_num=500` | 暫不照搬 | 原始碼設定來自 UGC benchmark；本專案先維持 top-k 10，GraphProxy 穩定後再做 sensitivity。 |
| Aggregate RAG columns | 不再作為主要逼近方向 | `skapp_proxy` 已是 aggregate proxy 的合理上限；再加欄位對 source alignment 幫助有限。 |
| Retrieved-set tensor artifacts | 已完成第一版，後續可強化 | `skapp_graph_proxy` 已輸出 selected retrieved text/image/label/contribution tensors；目前採固定 top-k padding。 |
| RRCP_silver | 已完成第一版，必須繼續 debug | `C3-ProjectInputSKAPPFull` 已用 all-items model + single/dissembled model 產生 RRCP_silver，但性能顯示訓練/正規化仍未調好。 |
| GraphLearner-style fusion | 已完成第一版，後續可強化 | `C3-ProjectInputSKAPPFull` 已使用 source-shaped cosine graph + graph convolution；仍需更細地對齊原始碼 hidden size/normalization。 |
| RRCP-Attention prediction head | 已完成第一版，後續可強化 | `C3-ProjectInputSKAPPFull` 已用 RRCP/CXMI-style weights 聚合 packed text/image graph outputs，再 concat retrieved label embedding。 |
| Social-media UGC context | 棄用 | 會換掉研究任務，不符合 anime pre-release input contract。 |
| 直接搬原始碼 | 棄用 | 原始碼 schema / CUDA / retrieval pool 設計不符合本專案；只能參考架構，不直接移植。 |

C3 row 決策：

| Row | 決策 |
|---|---|
| `C3-RAG-None-XGB` | 保留為 no-RAG control。 |
| `C3-RAG-Sparse-XGB` | 保留為 simple sparse retrieval row，不再優先延伸。 |
| `C3-RAG-Dense-XGB` | 保留為 semantic retrieval comparison，不再優先 tune。 |
| `C3-RAG-Hybrid-XGB` | 保留為 RRF comparison，不再主張 hybrid 必然更好。 |
| `C3-RAG-Selective-XGB` | 保留為 strongest performance row。 |
| `C3-ProjectInputSKAPPProxy-XGB` | 保留為 closest aggregate SKAPP-style proxy。 |
| `C3-ProjectInputSKAPPGraphProxy` | 保留為 closest architecture proxy；performance 不如 selective row。 |
| `C3-ProjectInputSKAPPFull` | 新增為正式 reconstruction 目標；完成前 C3 不能宣稱有完整 SKAPP 對比。 |

## 2026-05-19：C1/C2 structure-complete project-input reconstruction

本輪補齊 C1/C2 的主要缺口：不再只使用 MiniLM/Swin project embeddings
做 proxy，而是補上與原論文更接近的 encoder artifacts 和主要模型構造。

新增 artifact：

```text
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_train.parquet
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_val.parquet
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_test.parquet
```

產生指令：

```bash
python -m src.reference_baseline_branch.build_gpt2_text_embeddings --splits train val test --batch-size 16 --device auto --local-files-only
```

### C1-Armenta-ProjectInputReconstruction

執行指令：

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-ProjectInputReconstruction --include-disabled
```

Run：

```text
.exp/baseline/results/36
```

結構：

```text
metadata + GPT-2 synopsis embeddings + ResNet-50 cover/banner features
-> synopsis branch 768
-> project-context MLP 768
-> Armenta Big MLP 768,384,192,96,48,24,12,6
-> regression
```

結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|---:|
| `C1-Armenta-ProjectInputReconstruction` | popularity | 10719.7513 | 0.2898 | 0.8192 | 1.0563 |
| `C1-Armenta-ProjectInputReconstruction` | meanScore | 9.0250 | -0.1096 | 0.4666 |  |

判斷：

```text
這是目前 C1 主線中最接近原論文主要構造的 project-input reconstruction。
它補上 GPT-2 synopsis embedding 與 ResNet-50 visual features，但仍不使用
main-character descriptions / portraits，因此不可稱 exact reproduction。
性能上它不如 C1-Armenta-ProjectInputProxy-ResNet50；它的價值主要是論文
對齊，而不是最強 C1 performance。
```

### C1-Armenta-Figure2Reconstruction

執行指令：

```bash
python -m src.reference_baseline_branch.build_c1_character_features --splits train val test --batch-size 32 --portrait-batch-size 32 --download-workers 24 --max-characters 5 --device auto --local-files-only
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-Figure2Reconstruction --include-disabled
```

Run：

```text
.exp/baseline/results/38
```

結構：

```text
GPT-2 synopsis embeddings
+ GPT-2 main-character description/name embeddings
+ ResNet-50 main-character portrait features
-> Armenta character MLP Dropout/Linear/Tanh/Dropout/Linear
-> Armenta Big MLP 768,384,192,96,48,24,12,6
-> regression
```

Character artifact coverage：

| split | rows | has description | has portrait URL | encoded portrait |
|---|---:|---:|---:|---:|
| train | 9583 | 4755 | 5620 | 4984 |
| val | 2918 | 1415 | 1921 | 1718 |
| test | 3087 | 1578 | 2193 | 1931 |

結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|---:|
| `C1-Armenta-Figure2Reconstruction` | popularity | 11878.6328 | 0.3556 | 0.7823 | 1.1688 |
| `C1-Armenta-Figure2Reconstruction` | meanScore | 9.7747 | -0.2172 | 0.3824 |  |

判斷：

```text
這是目前最接近 Armenta-Segura & Sidorov 2025 Figure 2 的旁支復現：
它補上 character description/name 與 portrait branch，並保留原論文
character MLP + Big MLP 形狀。但因為它改用 character-specific artifacts，
不是本專案主框架的 cover/banner image input，因此不可當作主線 baseline。
缺 raw character/portrait 的 row 使用 zero-filled features；因此也不可宣稱
character coverage 與原論文一致。
```

### C2-ProjectInputCTNNReconstruction

執行指令：

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputCTNNReconstruction --include-disabled
```

Run：

```text
.exp/baseline/results/37
```

結構：

```text
metadata + GPT-2 synopsis embeddings + ResNet-50 cover/banner features
-> text transformer encoder over GPT-2 feature chunks
-> visual transformer encoder over cover/banner tokens
-> bidirectional text-image cross-modal attention
-> GRU recurrent fusion over text/image/metadata-factor tokens
-> metadata factor gate
-> regression
```

結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|---:|
| `C2-ProjectInputCTNNReconstruction` | popularity | 10151.2161 | 0.4608 | 0.8471 | 0.9981 |
| `C2-ProjectInputCTNNReconstruction` | meanScore | 8.1751 | 0.0696 | 0.5247 |  |

判斷：

```text
這是目前 C2 主線中最接近 CTNN 主要構造的 project-input reconstruction。
它也是目前 C2 中最佳的 row：popularity R2 高於 CrossAttention/RecurrentFusion，
meanScore R2 與 Spearman 也略高。仍不可稱 exact CTNN reproduction，因為原論文
使用 movie posters/reviews、box-office class/range target 與原始 movie dataset。
```

### C2-ProjectInputCTNNDualVisualReconstruction

執行指令：

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputCTNNDualVisualReconstruction --include-disabled
```

Run：

```text
.exp/baseline/results/39
```

結構：

```text
metadata + GPT-2 synopsis embeddings
+ project image embeddings as ViT-like cover/banner visual semantic tokens
+ ResNet-50 cover/banner features
-> text transformer encoder over GPT-2 feature chunks
-> visual transformer encoder over ResNet cover/banner + project-image cover/banner tokens
-> bidirectional text-image cross-modal attention
-> GRU recurrent fusion over text/image/metadata-factor tokens
-> metadata factor gate
-> regression
```

結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|---:|
| `C2-ProjectInputCTNNDualVisualReconstruction` | popularity | 10214.6356 | 0.4421 | 0.8491 | 0.9399 |
| `C2-ProjectInputCTNNDualVisualReconstruction` | meanScore | 8.8957 | -0.0720 | 0.5310 |  |

判斷：

```text
這是 C2 的 source-alignment diagnostic，不取代主線。
原論文 poster branch 同時使用 ResNet50 與 ViT；本版本在 project input contract
下以 ResNet-50 cover/banner features 加上既有 project image embeddings 作為
ViT-like semantic visual stream。結果顯示 popularity Spearman 與 log_MAE 小幅
優於單視覺 reconstruction，但 popularity R2 較低，meanScore R2 明顯變差。
因此目前 C2 主線仍是 C2-ProjectInputCTNNReconstruction；dual-visual row
主要用來證明我們已檢查 ResNet+ViT 對齊方向，而不是當最佳性能 row。
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

1. 若目標是補齊 reference map，目前 C3/SKAPP-inspired route 已可放進 baseline table。
2. 只有在目標變成提升性能，而不是補完整 reference map 時，才優先 tune 或替換 C1/C2 neural fusion heads。
3. 決定目前報告是否讓 `C3-RAG-Selective-XGB` 作為 RAG route 的 strongest reference row，並把 `F2-XGB-Concat` 作為 no-RAG multimodal classical floor。

如果目標從「cover the reference route」改成「在本專案輸入下更貼近原論文模型」，優先順序應改為：

1. `C1-Armenta-ProjectInputReconstruction` 作為 C1 project-input 主線：它對齊本專案 metadata / synopsis / cover-banner input contract，並補上 GPT-2 synopsis、ResNet-50 cover/banner、project-context MLP 與 Armenta Big MLP。
2. `C1-Armenta-Figure2Reconstruction` 保留為 non-mainline side analysis：它補上 main-character description/name 與 portrait branch，但輸入契約已不等同本專案 cover/banner 主框架。
3. `C2-ProjectInputCTNNReconstruction` 作為 C2 主線：它補上 modality transformer encoders、bidirectional cross-modal attention、GRU recurrent fusion 與 metadata factor gate。
4. C3 目前保留四個定位：`C3-RAG-Selective-XGB` 是 strongest performance row；`C3-ProjectInputSKAPPProxy-XGB` 是 aggregate SKAPP-style proxy；`C3-ProjectInputSKAPPGraphProxy` 是 closest architecture proxy；`C3-ProjectInputSKAPPFull` 是 first structure-complete reconstruction run，但需要 debug performance。
5. 若要繼續 C3，下一步不是再堆 aggregate 欄位，而是強化 full reconstruction：debug all-items/single-item training、RRCP_silver distribution、variable-length handling、以及更完整的 RRCP/CXMI attention head。
