# Reference Baseline Weekly Update 2026-05-26

## 1. 進度

### Baseline 目錄與版本基準

- 已將 reference baseline 與 ablation baseline 的定位分開：
  - `src/reference_baseline_branch/`：文獻復現與外部 baseline
  - `src/ablation_branch/`：我們自己框架的 ablation baseline
  - `src/experiment_common/`：共用 metrics 與 feature loader
- 已確認正式比較要以 V2 資料為基準：
  - `data/fussion/fusion_meta_clean_train_v2.csv`
  - `data/fussion/fusion_meta_clean_val_v2.csv`
  - `data/fussion/fusion_meta_clean_test_v2.csv`
- 已把 reference baseline config 切到 V2：
  - `data.meta_dir: data/fussion`
  - `data.meta_suffix: "_v2"`
- 已補上 V2 專用 artifact 路徑，避免舊 post2000 結果混入正式比較：
  - GPT-2 text artifact: `.exp/baseline/text_features/gpt2_v2`
  - ResNet-50 image artifact: `.exp/baseline/image_features/resnet50_v2`
  - C1 character artifact: `.exp/baseline/c1_character_features_v2`
  - C3 RAG artifact: `.exp/baseline/rag_features_v2`
- 2026-05-26 已完成必要 artifact：
  - V2 GPT-2 synopsis artifact：train 13,321 / val 2,918 / test 3,087
  - V2 ResNet-50 cover/banner artifact：train 13,321 / val 2,918 / test 3,087
  - V2 SKAPP graph proxy artifact：train 13,321 / val 2,918 / test 3,087

### 已完成的 V2 reference baseline 結果

已完成 20 個 baseline ID、共 40 筆 target 結果：

- Foundation / simple baseline：
  - `F0-Mean`
  - `F0-Ridge-Meta`
  - `F1-RF-Meta`
  - `F1-GB-Meta`
  - `F2-XGB-Concat`
  - `T2-XGB-TextEmb`
  - `I1-XGB-ImageEmb`
- C1 Armenta route：
  - `C1-Armenta-ProjectInputProxy`
  - `C1-Armenta-ProjectInputReconstruction`
- C2 Box-office / CTNN route：
  - `C2-CTNN-Lite`
  - `C2-ProjectInputCrossAttention`
  - `C2-ProjectInputRecurrentFusion`
  - `C2-ProjectInputCTNNReconstruction`
- C3 SKAPP / RAG route：
  - `C3-RAG-None-XGB`
  - `C3-RAG-Sparse-XGB`
  - `C3-RAG-Dense-XGB`
  - `C3-RAG-Hybrid-XGB`
  - `C3-RAG-Selective-XGB`
  - `C3-ProjectInputSKAPPProxy-XGB`
  - `C3-ProjectInputSKAPPGraphProxy`

目前可讀結果表：

- `reports/reference_baseline_v2_results.md`
- `reports/reference_baseline_v2_results.csv`
- `reports/reference_baseline_v2_vs_previous.csv`
- `reports/v2_input_effect_comparison.md`
- `reports/reference_baseline_paper_table_decision_2026-05-30.md`：論文主表/附錄採用決策，以這份為目前定稿依據

### 目前 V2 baseline 主要結果

| 類別 | baseline | target | test MAE | test R2 | test Spearman |
|---|---|---|---:|---:|---:|
| metadata baseline | `F1-RF-Meta` | popularity | 8551.7168 | 0.5865 | 0.8420 |
| metadata baseline | `F1-RF-Meta` | meanScore | 8.0179 | 0.1111 | 0.5759 |
| multimodal concat | `F2-XGB-Concat` | popularity | 9688.3006 | 0.5108 | 0.8579 |
| multimodal concat | `F2-XGB-Concat` | meanScore | 8.5473 | -0.0231 | 0.5102 |
| C1 proxy | `C1-Armenta-ProjectInputProxy` | popularity | 11672.2261 | 0.3794 | 0.8287 |
| C1 proxy | `C1-Armenta-ProjectInputProxy` | meanScore | 9.2523 | -0.1983 | 0.4307 |
| C1 reconstruction | `C1-Armenta-ProjectInputReconstruction` | popularity | 10501.5398 | 0.3963 | 0.8149 |
| C1 reconstruction | `C1-Armenta-ProjectInputReconstruction` | meanScore | 10.5367 | -0.4982 | 0.4447 |
| C2 proxy | `C2-ProjectInputCrossAttention` | popularity | 11044.7140 | 0.4165 | 0.8473 |
| C2 proxy | `C2-ProjectInputCrossAttention` | meanScore | 8.4384 | 0.0087 | 0.4837 |
| C2 reconstruction | `C2-ProjectInputCTNNReconstruction` | popularity | 10448.2886 | 0.4189 | 0.8481 |
| C2 reconstruction | `C2-ProjectInputCTNNReconstruction` | meanScore | 8.3066 | 0.0541 | 0.5269 |
| C3 RAG | `C3-RAG-Selective-XGB` | popularity | 9520.2222 | 0.5901 | 0.8719 |
| C3 RAG | `C3-ProjectInputSKAPPProxy-XGB` | meanScore | 8.2630 | 0.0472 | 0.5217 |
| C3 graph proxy | `C3-ProjectInputSKAPPGraphProxy` | popularity | 11512.0077 | 0.4046 | 0.8563 |
| C3 graph proxy | `C3-ProjectInputSKAPPGraphProxy` | meanScore | 8.5741 | -0.0355 | 0.4719 |

### V2 與前一版 baseline 的比較

- 舊 baseline 多數是以 `data/fussion/post2000` 作為 train set。
- V2 baseline 改為 full V2 train set：
  - post2000 train：9,583 筆
  - V2 train：13,321 筆
  - 多出 3,738 筆 1940-1999 年作品
- V2 CSV 相對 full 原始 CSV 只移除無真實封面圖/default 圖資料：
  - train 移除 55 筆
  - val/test 不變
  - holdout_unknown 移除 2 筆

因此目前「V2 vs previous」的差異主要來自 train set 從 post2000 擴大為 full V2，而不是單純新版 image embedding 的效果。

## 2. 難點

### V2 的定義容易混淆

目前 dev 合進來的 V2 包含兩件事：

- `fusion_meta_clean_{split}_v2.csv`
- `src_2/model/best` 與 checkpoint

但目前 baseline 實際使用的是：

- V2 metadata/split
- 舊的 project image embedding parquet：
  - `src/fussion_branch/embedding/image/image_embeddings_train.parquet`
  - `src/fussion_branch/embedding/image/image_embeddings_val.parquet`
  - `src/fussion_branch/embedding/image/image_embeddings_test.parquet`

目前沒有任何程式碼自動引用 `src_2`。所以目前結果不能宣稱是「使用 `src_2` 新 image encoder 的 V2 image baseline」。

### C1/C2/C3 的必要外部 baseline 已補齊

原本待補的三條必要主線已完成 V2 artifact 與 run：

| 路線 | run | 定位 |
|---|---|---|
| `C1-Armenta-ProjectInputReconstruction` | `.exp/baseline/results/v2_01_12` | C1 外部 baseline 主線；GPT-2 synopsis + ResNet-50 cover/banner + Armenta-shaped Big MLP |
| `C2-ProjectInputCTNNReconstruction` | `.exp/baseline/results/v2_01_13` | C2 外部 baseline 主線；modality transformer encoder + cross attention + recurrent fusion + metadata gate |
| `C3-ProjectInputSKAPPGraphProxy` | `.exp/baseline/results/v2_01_14` | C3 architecture proxy；retrieved tensors + RRCP-style mask + graph/attention context learning |

仍需保留的限制：

- 這三條可以寫成 external / adapted SOTA baseline。
- 不能寫成 exact reproduction，因為資料集、target、split 與原論文仍不同。
- C1/C2 是 project-input reconstruction，不是原資料任務的數值復現。
- C3 GraphProxy 比 SKAPPProxy 更像架構，但仍不是完整 SKAPP/RRCP/VL-GNN 原始碼級復現。

剩下未做的項目改列 optional / appendix：

- `C1-Armenta-Figure2Reconstruction`：偏 character-centric side reconstruction，不適合作為本專案主表。
- `C2-ProjectInputCTNNDualVisualReconstruction`：需要先決定是否用 `src_2` 重產 project Swin image stream。
- `C3-ProjectInputSKAPPFull`：舊版已跑通但性能弱；V2 full runner 成本高，先不擋主線。

### 目前 baseline 的 image modality 還不是最終 V2 image

目前 baseline image modality 的 ID 對齊是完整的：

| split | V2 metadata | image parquet | intersection |
|---|---:|---:|---:|
| train | 13,321 | 13,376 | 13,321 |
| val | 2,918 | 2,918 | 2,918 |
| test | 3,087 | 3,087 | 3,087 |

但 image parquet 產生時間是 2026-04-27，不是 dev 合進來的 `src_2` 模型重新產生的結果。

因此目前 V2 baseline 可視為：

> V2 metadata/split + 現有 project image embeddings

不能視為：

> V2 metadata/split + `src_2` 新 Swin model image embeddings

### Fusion 自身結果與 README 敘述不一致

`src/fussion_branch/README.md` 中 V2 對應 Run15，但表格數字顯示：

- popularity test log_MAE：Run11 0.9766，Run15 1.0112
- meanScore test MAE：Run11 8.0691，Run15 8.0865

依表格看 Run15 沒有明顯優於 Run11；README 文字「Run15 在兩個 target 的 val 均達最佳」與表格數字不一致，需要後續確認或修正。

## 3. 預期進度

### 缺口預估耗時與之前未跑原因

| 缺口 | 需要做什麼 | 粗估耗時 | 前幾次未跑原因 |
|---|---|---:|---|
| V2 project image embeddings | 確認 `src_2/model/best` 是否為正式 image encoder，並用它重產 `image_embeddings_{split}.parquet` | 約 30 分鐘到 2 小時，視 GPU/CPU 與圖片路徑狀況 | 當時還沒確認 `src_2` 來源與定位；若直接重跑，可能產出一份不知道能不能正式引用的 image artifact |
| C1/C2 V2 GPT-2 text artifact | 跑 `build_gpt2_text_embeddings.py`，輸出 `.exp/baseline/text_features/gpt2_v2` | 已完成，約 28 分鐘 | 之前先用現有 project text embedding 做 project-input proxy，因為正式比較先需要一張可跑完的 V2 baseline 表；GPT-2 artifact 屬於完整 reconstruction，不是第一張表的必要條件 |
| C1/C2 V2 ResNet-50 artifact | 跑 `build_resnet50_image_embeddings.py`，輸出 `.exp/baseline/image_features/resnet50_v2` | 已完成，約 15 分鐘 | 同上；而且 image pipeline 是否要改用 `src_2` 仍未確認，若先跑 ResNet-50 仍無法解決 V2 image encoder 是否一致的問題 |
| C1 Figure2 character artifact | 跑 `build_c1_character_features.py`，包含 character descriptions 與 portraits | 1-3 小時以上；若 portrait URL 需下載，可能更久且受網路影響 | 這條路線與我們主框架輸入不完全對標，只適合作為 side reconstruction；之前決定先做 project-input proxy，避免產出一個很完整但無法當主比較的結果 |
| C1 ProjectInputReconstruction | 使用 V2 GPT-2 + V2 ResNet-50 artifact 跑 Armenta-shaped Big MLP | 已完成，約 7 分鐘 | 原本缺 V2 GPT-2/ResNet-50 artifact；現在已補齊並跑出 `.exp/baseline/results/v2_01_12` |
| C2 ProjectInputCTNNReconstruction | 使用 V2 GPT-2 + V2 ResNet-50 artifact 跑 CTNN reconstruction | 已完成，約 10 分鐘 | 原本缺 V2 GPT-2/ResNet-50 artifact；現在已補齊並跑出 `.exp/baseline/results/v2_01_13` |
| C2 DualVisualReconstruction | 使用 V2 GPT-2 + V2 ResNet-50 + project image embedding 跑 dual visual CTNN | artifact 完成後約 30-90 分鐘 | 需要先決定 project image embedding 是否重產為 `src_2` 版本，否則 dual visual 的 image stream 定義不穩 |
| C3 skapp_graph_proxy artifact | 建立非常寬的 graph/RRCP feature parquet，再跑 `C3-ProjectInputSKAPPGraphProxy` | 已完成，artifact 約 27 分鐘；model 約 3 分鐘 | 原本因成本與寬表風險排後；現在已補齊並跑出 `.exp/baseline/results/v2_01_14` |
| C3 SKAPPFull V2 | 跑 `run_c3_skapp_full.py`，包含 all-items model、single-item model、RRCP_silver、final model | CPU 可能 2-6 小時以上；GPU 也需較長時間 | 舊版結果表現偏差且訓練鏈條長；在 V2 image 與 RAG artifact 未定前，先跑會很難解釋 |

總結：之前不是因為這些缺口不能跑，而是因為當時有三個前提沒定：

1. 正式比較基準是否改為 V2
2. V2 image 是否要使用 `src_2` 重新產生 embedding
3. C1/C2/C3 應先做 project-input proxy，還是先跑成本高、但不一定可作主比較的完整 reconstruction

現在第 1 點已確定為 V2；第 2 點仍需確認；第 3 點已先完成 project-input proxy 與 C3 XGB RAG baseline。

### 短期：先把正式週會表格定稿

- 用 `reports/reference_baseline_v2_results.md` 作為本週 reference baseline 結果表。
- 會議中明確說明：
  - 目前 V2 baseline 是 V2 metadata/split 對齊結果
  - 目前尚未使用 `src_2` 重新產生 image embedding
  - 舊 baseline 與 V2 baseline 的主要差異是 train set 從 post2000 擴成 full V2

### 接下來第一優先：確認 `src_2` 的定位

需要確認：

- `src_2/model/best` 是否就是新版 image encoder
- `src_2` 是否應搬到正式路徑，例如：
  - `src/fussion_branch/model/best`
  - 或 `.exp/image_encoder/v2/best`
- `src_2/fussion_configs.yaml` 為何是空檔
- 是否要用 `src_2/model/best` 重新產生 V2 image embeddings

如果確認要用，預期產出：

- `image_embeddings_train_v2.parquet`
- `image_embeddings_val_v2.parquet`
- `image_embeddings_test_v2.parquet`

或獨立目錄：

- `src/fussion_branch/embedding/image_v2/image_embeddings_{split}.parquet`

### 接下來第二優先：重跑 image-dependent baseline

若完成新版 V2 image embeddings，需重跑：

- `I1-XGB-ImageEmb`
- `F2-XGB-Concat`
- `C1-Armenta-ProjectInputProxy`
- `C2-CTNN-Lite`
- `C2-ProjectInputCrossAttention`
- `C2-ProjectInputRecurrentFusion`
- C3 所有包含 image 的 RAG baseline

重跑後才能正式討論：

> V2 image encoder / extraLarge cover 是否真的改善 baseline 與主框架效果

### 接下來第三優先：C1/C2/C3 reconstruction 的收尾

必要主線已完成，接下來不是繼續擴 baseline 數量，而是收斂論文可寫法：

1. 將 `C1-Armenta-ProjectInputReconstruction` 寫成 C1 adapted external baseline 主線。
2. 將 `C2-ProjectInputCTNNReconstruction` 寫成 C2 adapted external baseline 主線。
3. 將 `C3-RAG-Selective-XGB` 寫成 C3 strongest performance RAG row。
4. 將 `C3-ProjectInputSKAPPGraphProxy` 寫成 C3 closest architecture proxy，而不是 strongest row。
5. `C1-Figure2`、`C2-DualVisual`、`C3-SKAPPFull` 只在需要 appendix 或老師要求更完整復現時再跑。

### 預期下週可交付

- 一份正式 V2 reference baseline table，已更新到 40 筆 target 結果
- 一份 V2 vs previous baseline comparison，已納入 C1/C2/C3 reconstruction rows
- 確認 `src_2` 是否納入正式 image embedding pipeline
- 若 `src_2` 可用，完成新版 V2 image embeddings
- 至少重跑 image-dependent baseline 的第一批結果
