# Reference Baseline Paper Table Decision 2026-05-30

本文整理 C1/C2/C3 與基礎 baseline 在成果論文中的採用方式。目的不是再增加路線，而是固定主表、附錄、宣稱邊界與 raw result 保存策略。

## 1. 主表採用名單

正式主表建議放以下 rows。這些 rows 都已使用 V2 comparison contract：

- `data/fussion/fusion_meta_clean_train_v2.csv`
- `data/fussion/fusion_meta_clean_val_v2.csv`
- `data/fussion/fusion_meta_clean_test_v2.csv`

| 論文角色 | baseline_id | target | test_MAE | test_R2 | test_Spearman | run_dir | 採用理由 |
|---|---|---:|---:|---:|---:|---|---|
| Mean floor | `F0-Mean` | popularity | 14935.1 | -0.1479 | 0.0000 | `.exp/baseline/results/v2_01` | 最低限度常數預測 |
| Mean floor | `F0-Mean` | meanScore | 10.9115 | -0.4631 | 0.0000 | `.exp/baseline/results/v2_01` | 最低限度常數預測 |
| Metadata classical baseline | `F1-RF-Meta` | popularity | 8551.7168 | 0.5865 | 0.8420 | `.exp/baseline/results/v2_01` | 目前 metadata-only 最強 row |
| Metadata classical baseline | `F1-RF-Meta` | meanScore | 8.0179 | 0.1111 | 0.5759 | `.exp/baseline/results/v2_01` | 目前 metadata-only 最強 row |
| Multimodal classical baseline | `F2-XGB-Concat` | popularity | 9688.3006 | 0.5108 | 0.8579 | `.exp/baseline/results/v2_01` | Metadata + text + image 的簡潔 multimodal floor |
| Multimodal classical baseline | `F2-XGB-Concat` | meanScore | 8.5473 | -0.0231 | 0.5102 | `.exp/baseline/results/v2_01` | Metadata + text + image 的簡潔 multimodal floor |
| C1 adapted external baseline | `C1-Armenta-ProjectInputReconstruction` | popularity | 10501.5398 | 0.3963 | 0.8149 | `.exp/baseline/results/v2_01_12` | 補齊 GPT-2 synopsis、ResNet-50 cover/banner、Armenta-shaped Big MLP |
| C1 adapted external baseline | `C1-Armenta-ProjectInputReconstruction` | meanScore | 10.5367 | -0.4982 | 0.4447 | `.exp/baseline/results/v2_01_12` | 同上，但 score 預測不穩，需如實報告 |
| C2 adapted external baseline | `C2-ProjectInputCTNNReconstruction` | popularity | 10448.2886 | 0.4189 | 0.8481 | `.exp/baseline/results/v2_01_13` | 補齊 transformer encoders、cross attention、GRU recurrent fusion、metadata gate |
| C2 adapted external baseline | `C2-ProjectInputCTNNReconstruction` | meanScore | 8.3066 | 0.0541 | 0.5269 | `.exp/baseline/results/v2_01_13` | C2 主線 reconstruction，比 CTNN-Lite 更適合放主表 |
| C3 strongest RAG baseline | `C3-RAG-Selective-XGB` | popularity | 9520.2222 | 0.5901 | 0.8719 | `.exp/baseline/results/v2_01_10` | C3 route 目前最強 performance row |
| C3 strongest RAG baseline | `C3-RAG-Selective-XGB` | meanScore | 8.3090 | 0.0418 | 0.5234 | `.exp/baseline/results/v2_01_10` | C3 route 目前最強 performance row |
| C3 architecture proxy baseline | `C3-ProjectInputSKAPPGraphProxy` | popularity | 11512.0077 | 0.4046 | 0.8563 | `.exp/baseline/results/v2_01_14` | 最接近 SKAPP retrieved-set tensor + graph/attention 結構的 project-input proxy |
| C3 architecture proxy baseline | `C3-ProjectInputSKAPPGraphProxy` | meanScore | 8.5741 | -0.0355 | 0.4719 | `.exp/baseline/results/v2_01_14` | 架構對齊價值高，但 performance 不是最強 |

## 2. 不放主表但可放附錄或方法說明

| baseline_id | 建議定位 | 原因 |
|---|---|---|
| `F0-Ridge-Meta` | appendix / sanity check | 線性 metadata floor，可保留但不必佔主表 |
| `F1-GB-Meta` | appendix | 與 `F1-RF-Meta` 同類，主表留較強且穩定的 RF 即可 |
| `T2-XGB-TextEmb` | appendix / modality-only note | 單文字 embedding row，可支援「文字單模態不足」觀察 |
| `I1-XGB-ImageEmb` | appendix / modality-only note | 單影像 embedding row，可支援「影像單模態不足」觀察 |
| `C1-Armenta-ProjectInputProxy` | milestone | 已被 `C1-Armenta-ProjectInputReconstruction` 取代 |
| `C2-CTNN-Lite` | milestone | 太簡化，不應作為 C2 主線 |
| `C2-ProjectInputCrossAttention` | milestone / ablation-like proxy | 已被 `C2-ProjectInputCTNNReconstruction` 取代 |
| `C2-ProjectInputRecurrentFusion` | milestone / ablation-like proxy | 已被 `C2-ProjectInputCTNNReconstruction` 取代 |
| `C3-RAG-None/Sparse/Dense/Hybrid` | C3 internal comparison | 可用於說明 retrieval strategy，但主表可只放 strongest selective |
| `C3-ProjectInputSKAPPProxy-XGB` | aggregate SKAPP proxy | 比 graph proxy 更快、更穩，但架構對齊度低於 graph proxy |

## 3. 目前不建議阻塞主線的項目

| 項目 | 狀態 | 決策 |
|---|---|---|
| `C1-Armenta-Figure2Reconstruction` | 未跑 V2 character artifact | 不阻塞主表；它更貼近 Figure 2 character branch，但不對齊本專案 cover/banner project input |
| `C2-ProjectInputCTNNDualVisualReconstruction` | 未跑 V2 | 不阻塞主表；需先決定 project Swin stream 是否改用 `src_2` 重產 embedding |
| `C3-ProjectInputSKAPPFull` V2 | 未重跑 | 不阻塞主表；舊版完整 runner 已跑通但 performance 弱，V2 full run 適合作為後續診斷 |
| `src_2` image encoder 接入 | 未接進 baseline pipeline | 不阻塞 external baseline；但若要宣稱新版 image encoder 效果，必須另開 image V2 experiment |

## 4. 論文可用宣稱

建議用語：

- C1：`Adapted Armenta-style project-input reconstruction`
- C2：`Adapted CTNN-style project-input reconstruction`
- C3：`Selective retrieval baseline` 與 `SKAPP-style graph proxy`

可宣稱：

- 已使用同一份 V2 train/val/test split 與相同 evaluation metrics。
- C1/C2 已補齊 GPT-2 synopsis artifact 與 ResNet-50 visual artifact。
- C2 主線已包含 CTNN-style modality encoder、cross-modal attention、recurrent fusion 與 metadata conditioning。
- C3 GraphProxy 已包含 retrieved-set tensor、RRCP-style mask、learned graph adjacency 與 contribution-aware attention。

不可宣稱：

- 不可稱為 exact reproduction。
- 不可宣稱與原論文數值可直接比較。
- 不可宣稱 C1 使用完整原論文 main-character description/portrait contract。
- 不可宣稱 C2 使用 movie review/poster/box-office original task。
- 不可宣稱 C3 完整重現 SKAPP 的 social-media UGC、RRCP pretraining、VL-GNN 與完整 RRCP-Attention pipeline。
- 不可宣稱目前 baseline 已使用 `src_2` 新 Swin image encoder。

## 5. Raw result 保存策略

目前 `.exp/` 被 `.gitignore` 忽略，因此 raw run directory 不會自動提交。

建議提交：

- `reports/baselines/reference_baseline_v2_results.csv`
- `reports/baselines/reference_baseline_v2_results.md`
- `reports/baselines/reference_baseline_v2_vs_previous.csv`
- `reports/baselines/reference_baseline_v2_vs_previous_comparison.csv`
- `reports/baselines/reference_baseline_v2_vs_previous_comparison.md`
- `reports/baselines/reference_baseline_weekly_update_2026-05-26.md`
- `reports/experiments/v2_input_effect_comparison.md`
- `reports/baselines/reference_baseline_paper_table_decision_2026-05-30.md`

不建議提交：

- `.exp/baseline/text_features/gpt2_v2/*.parquet`
- `.exp/baseline/image_features/resnet50_v2/*.parquet`
- `.exp/baseline/rag_features_v2/**/*.parquet`
- `.exp/baseline/results/**/predictions/*.csv`

可選擇 force-add 的小型 raw evidence：

- `.exp/baseline/results/v2_01_12/baseline_results.csv`
- `.exp/baseline/results/v2_01_13/baseline_results.csv`
- `.exp/baseline/results/v2_01_14/baseline_results.csv`
- `.exp/baseline/results/v2_01_12/config.yaml`
- `.exp/baseline/results/v2_01_13/config.yaml`
- `.exp/baseline/results/v2_01_14/config.yaml`

若要讓其他組員完全重跑，應以 reports + config + reconstruction commands 為主，而不是提交大型 parquet artifact。

## 6. 建議提交批次

| 批次 | 內容 | 建議 commit type |
|---|---|---|
| 1 | baseline feature loader、artifact builders、runner/config 的 V2 與 reconstruction 支援 | `feat(baseline)` |
| 2 | V2 reference baseline result CSV/MD 與 previous comparison | `docs(baseline)` |
| 3 | paper table decision、weekly update、V2 interpretation docs | `docs(reports)` |
| 4 | 如決定保留小型 raw evidence，force-add run 12-14 的 `baseline_results.csv` 與 `config.yaml` | `chore(results)` |

不應混入同一批：

- EDA feature filter files
- presentation materials
- unrelated proposal PDFs
- `src/fussion_branch/run_rag_ablation.py`
- `src/fussion_branch/run_rq2_rag_experiments.py`
