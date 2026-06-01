# 2026-05-19 SOTA / Reference Paper Reconstruction Summary

## 會議結論

本週最大的成果不是完成正式 EXP1/EXP2，而是把三條文獻 / SOTA
reference routes 補到可報告的 reconstruction 狀態：

1. `C1` Anime popularity deep fusion route
2. `C2` CTNN cross-modal transformer route
3. `C3` SKAPP selective retrieval route

準確說法：

```text
我們已完成三條 SOTA/reference paper 的 project-input reconstruction 或
side reconstruction，並建立可比較的結果表。

這些結果是 EXP1/EXP2 的 reference baseline layer，
不是正式 project fusion EXP1/EXP2 ablation 本體。
```

不可說：

```text
We exactly reproduce the original papers.
```

因為三篇原論文的 dataset、target、split、部分 input source 都與本專案不同。
本專案採用的是 project-input reconstruction：在不破壞 anime pre-release input
contract 的前提下，盡量還原原論文主要模型構造。

## 最終 SOTA Reconstruction 結果總表

### Popularity

| Route | Final row | Role | test_MAE | test_R2 | Spearman | log_MAE | 判斷 |
|---|---|---|---:|---:|---:|---:|---|
| C1 Armenta | `C1-Armenta-ProjectInputReconstruction` | C1 主線 reconstruction | 10719.7513 | 0.2898 | 0.8192 | 1.0563 | 對齊主框架，但性能不是最強 |
| C1 Armenta | `C1-Armenta-Figure2Reconstruction` | C1 Figure 2 旁支 | 11878.6328 | 0.3556 | 0.7823 | 1.1688 | 更貼近 Figure 2，但不是主輸入框架 |
| C2 CTNN | `C2-ProjectInputCTNNReconstruction` | C2 主線 reconstruction | 10151.2161 | 0.4608 | 0.8471 | 0.9981 | C2 主線；R2 最佳 |
| C2 CTNN | `C2-ProjectInputCTNNDualVisualReconstruction` | C2 dual-visual diagnostic | 10214.6356 | 0.4421 | 0.8491 | 0.9399 | Spearman/log_MAE 較好，但 R2 較低 |
| C3 SKAPP | `C3-RAG-Selective-XGB` | C3 performance row | 9782.2338 | 0.5775 | 0.8746 | 0.9462 | 目前 RAG route 最強 |
| C3 SKAPP | `C3-ProjectInputSKAPPProxy-XGB` | aggregate SKAPP proxy | 10239.2909 | 0.5170 | 0.8574 | 0.9704 | learned contribution proxy |
| C3 SKAPP | `C3-ProjectInputSKAPPGraphProxy` | architecture proxy | 11501.8681 | 0.4404 | 0.8561 | 1.0245 | graph/attention proxy |
| C3 SKAPP | `C3-ProjectInputSKAPPFull` | full reconstruction diagnostic | 14668.1228 | -0.4927 | 0.6985 | 1.2983 | 架構跑通，但性能需 debug |

### MeanScore

| Route | Final row | Role | test_MAE | test_R2 | Spearman | 判斷 |
|---|---|---|---:|---:|---:|---|
| C1 Armenta | `C1-Armenta-ProjectInputReconstruction` | C1 主線 reconstruction | 9.0250 | -0.1096 | 0.4666 | 對 score 不穩 |
| C1 Armenta | `C1-Armenta-Figure2Reconstruction` | C1 Figure 2 旁支 | 9.7747 | -0.2172 | 0.3824 | 不適合作為 score 主 row |
| C2 CTNN | `C2-ProjectInputCTNNReconstruction` | C2 主線 reconstruction | 8.1751 | 0.0696 | 0.5247 | C2 主線，R2 較佳 |
| C2 CTNN | `C2-ProjectInputCTNNDualVisualReconstruction` | C2 dual-visual diagnostic | 8.8957 | -0.0720 | 0.5310 | Spearman 較佳，但 R2 較差 |
| C3 SKAPP | `C3-RAG-Selective-XGB` | C3 performance row | 8.0914 | 0.0905 | 0.5470 | 目前 RAG route 最強 |
| C3 SKAPP | `C3-ProjectInputSKAPPProxy-XGB` | aggregate SKAPP proxy | 8.1715 | 0.0744 | 0.5369 | 接近 selective，但略低 |
| C3 SKAPP | `C3-ProjectInputSKAPPGraphProxy` | architecture proxy | 8.1448 | 0.0690 | 0.4973 | MAE 尚可，rank 較弱 |
| C3 SKAPP | `C3-ProjectInputSKAPPFull` | full reconstruction diagnostic | 9.8063 | -0.2385 | 0.3657 | 架構跑通，但性能需 debug |

## C1：Armenta Anime Deep Fusion Route

對應論文：

```text
Anime popularity prediction before huge investments:
a multimodal approach using deep learning
```

### 完成狀態

已完成兩個層次：

| Row | 定位 | 是否主線 | 完成內容 |
|---|---|---|---|
| `C1-Armenta-ProjectInputReconstruction` | project-input reconstruction | 是 | GPT-2 synopsis + ResNet-50 cover/banner + project-context MLP + Big MLP |
| `C1-Armenta-Figure2Reconstruction` | Figure 2 side reconstruction | 否 | GPT-2 synopsis + main-character description/name + ResNet-50 portrait + character MLP + Big MLP |

### 為什麼有兩個 C1

`C1-Armenta-ProjectInputReconstruction` 保留本專案主框架輸入：

```text
metadata + synopsis/text + cover/banner image
```

所以它是 C1 主線。

`C1-Armenta-Figure2Reconstruction` 更接近原論文 Figure 2，但它改用：

```text
main-character descriptions + main-character portraits
```

這已經偏離本專案 cover/banner 主輸入契約，因此只能作旁支分析。

### C1 結果

| Row | target | test_MAE | test_R2 | Spearman | log_MAE |
|---|---:|---:|---:|---:|---:|
| `C1-Armenta-ProjectInputReconstruction` | popularity | 10719.7513 | 0.2898 | 0.8192 | 1.0563 |
| `C1-Armenta-ProjectInputReconstruction` | meanScore | 9.0250 | -0.1096 | 0.4666 |  |
| `C1-Armenta-Figure2Reconstruction` | popularity | 11878.6328 | 0.3556 | 0.7823 | 1.1688 |
| `C1-Armenta-Figure2Reconstruction` | meanScore | 9.7747 | -0.2172 | 0.3824 |  |

### C1 可主張

```text
C1 的主要模型構造已完成 project-input reconstruction。
另外也完成 Figure 2 character-centric side reconstruction。
```

### C1 不可主張

```text
不可說 exact reproduction。
不可說 Figure 2 row 是本專案主框架 baseline。
不可說 C1 性能已優於 F2/C3。
```

## C2：CTNN Cross-Modal Transformer Route

對應論文：

```text
Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers
```

### 完成狀態

| Row | 定位 | 是否主線 | 完成內容 |
|---|---|---|---|
| `C2-ProjectInputCTNNReconstruction` | project-input CTNN reconstruction | 是 | GPT-2 synopsis + ResNet-50 visual + transformer encoders + cross-attention + GRU recurrent fusion + metadata gate |
| `C2-ProjectInputCTNNDualVisualReconstruction` | dual-visual diagnostic | 否 | 在 C2 主線上加入 project image embeddings 作為 ViT-like visual stream |

### 為什麼有 dual-visual diagnostic

原 CTNN 視覺側包含：

```text
ResNet50 + ViT poster features
```

本專案沒有 MovieNet-finetuned ViT poster features，因此使用：

```text
ResNet-50 cover/banner features + project image embeddings
```

作為 ResNet50 + ViT-like visual proxy。它用來確認這個對齊方向是否有價值，
不是替代主線。

### C2 結果

| Row | target | test_MAE | test_R2 | Spearman | log_MAE |
|---|---:|---:|---:|---:|---:|
| `C2-ProjectInputCTNNReconstruction` | popularity | 10151.2161 | 0.4608 | 0.8471 | 0.9981 |
| `C2-ProjectInputCTNNReconstruction` | meanScore | 8.1751 | 0.0696 | 0.5247 |  |
| `C2-ProjectInputCTNNDualVisualReconstruction` | popularity | 10214.6356 | 0.4421 | 0.8491 | 0.9399 |
| `C2-ProjectInputCTNNDualVisualReconstruction` | meanScore | 8.8957 | -0.0720 | 0.5310 |  |

### C2 可主張

```text
C2 已完成 structure-complete project-input CTNN reconstruction。
Dual-visual diagnostic 已檢查 ResNet50 + ViT-like visual route。
```

### C2 不可主張

```text
不可說 exact CTNN reproduction。
不可說 dual-visual row 是新的主線。
不可說已使用 movie reviews / movie posters / box-office class setup。
```

## C3：SKAPP Selective Retrieval Route

對應論文：

```text
Improving Multimodal Social Media Popularity Prediction via
Selective Retrieval Knowledge Augmentation
```

### 完成狀態

目前 C3 有三種定位：

| Row | 定位 | 是否主線 | 說明 |
|---|---|---|---|
| `C3-RAG-Selective-XGB` | performance row | 是，目前最強 | RAG ablation 中表現最佳 |
| `C3-ProjectInputSKAPPProxy-XGB` | aggregate SKAPP proxy | 旁支 | learned contribution + attention-weighted aggregate features |
| `C3-ProjectInputSKAPPGraphProxy` | architecture proxy | 旁支 | retrieved-set tensor + RRCP-style mask + graph/attention |
| `C3-ProjectInputSKAPPFull` | full reconstruction diagnostic | 診斷 | all-items/single-item/RRCP_silver/GraphLearner-style full pipeline 已跑通，但性能差 |

### C3 結果

| Row | target | test_MAE | test_R2 | Spearman | log_MAE |
|---|---:|---:|---:|---:|---:|
| `C3-RAG-Selective-XGB` | popularity | 9782.2338 | 0.5775 | 0.8746 | 0.9462 |
| `C3-RAG-Selective-XGB` | meanScore | 8.0914 | 0.0905 | 0.5470 |  |
| `C3-ProjectInputSKAPPProxy-XGB` | popularity | 10239.2909 | 0.5170 | 0.8574 | 0.9704 |
| `C3-ProjectInputSKAPPProxy-XGB` | meanScore | 8.1715 | 0.0744 | 0.5369 |  |
| `C3-ProjectInputSKAPPGraphProxy` | popularity | 11501.8681 | 0.4404 | 0.8561 | 1.0245 |
| `C3-ProjectInputSKAPPGraphProxy` | meanScore | 8.1448 | 0.0690 | 0.4973 |  |
| `C3-ProjectInputSKAPPFull` | popularity | 14668.1228 | -0.4927 | 0.6985 | 1.2983 |
| `C3-ProjectInputSKAPPFull` | meanScore | 9.8063 | -0.2385 | 0.3657 |  |

### C3 可主張

```text
C3-RAG variants 已可作為 EXP2 reference baseline / sanity-check layer。
C3-RAG-Selective-XGB 是目前 RAG route 最強 performance row。
C3-ProjectInputSKAPPFull 已跑通完整構造，但還需要 debug/tuning。
```

### C3 不可主張

```text
不可說已 exact reproduce SKAPP。
不可用 SKAPPFull 的差性能說明 RAG 無效。
不可把 C3-RAG reference baseline 當成本專案主 fusion model 的 EXP2 已完成。
```

## 最終建議放進會議的版本

### 這週成果

```text
本週最大成果是完成 reference/SOTA baseline reconstruction layer：

1. C1 Armenta route：
   - project-input reconstruction 完成
   - Figure 2 character-centric side reconstruction 完成

2. C2 CTNN route：
   - project-input CTNN reconstruction 完成
   - ResNet50 + ViT-like dual-visual diagnostic 完成

3. C3 SKAPP route：
   - RAG none/sparse/dense/hybrid/selective variants 完成
   - SKAPP proxy / graph proxy / full reconstruction diagnostic 完成

這些結果可以支撐 EXP1/EXP2 的 baseline 對照，但正式 EXP1/EXP2 還需要接回我們自己的 fusion model。
```

### 最值得展示的數字

| 目的 | Row | popularity R2 | popularity Spearman | meanScore R2 | meanScore Spearman |
|---|---|---:|---:|---:|---:|
| Metadata floor | `F1-RF-Meta` | 0.5811 | 0.8466 | 0.1298 | 0.5836 |
| No-RAG multimodal floor | `F2-XGB-Concat` | 0.5194 | 0.8575 | 0.0193 | 0.5292 |
| C1 reconstruction | `C1-Armenta-ProjectInputReconstruction` | 0.2898 | 0.8192 | -0.1096 | 0.4666 |
| C2 reconstruction | `C2-ProjectInputCTNNReconstruction` | 0.4608 | 0.8471 | 0.0696 | 0.5247 |
| Best RAG reference | `C3-RAG-Selective-XGB` | 0.5775 | 0.8746 | 0.0905 | 0.5470 |
| SKAPP full diagnostic | `C3-ProjectInputSKAPPFull` | -0.4927 | 0.6985 | -0.2385 | 0.3657 |

會議解讀：

```text
目前最佳 reference performance 仍是 classical/RAG side：
F1-RF-Meta 與 C3-RAG-Selective-XGB 很強。

C1/C2 reconstruction 的價值主要是 SOTA paper alignment，不是目前最佳性能。
C3-SKAPPFull 的價值是完整構造跑通，但 performance 尚未可用。
```

## 下一步

1. 把 SOTA/reference baseline 表獨立放一節，不混入正式 EXP1/EXP2。
2. 正式 EXP1 要補同一主模型下的 modality ablation：
   - Metadata
   - Metadata + Text
   - Metadata + Image
   - Metadata + Text + Image
   - Metadata + RAG
   - Metadata + Text + RAG
   - Metadata + Image + RAG
   - Metadata + Text + Image + RAG
3. 正式 EXP2 要接回 project fusion model：
   - No-RAG
   - metadata RAG
   - text RAG
   - Hybrid RAG
   - Selective RAG
4. C3-SKAPPFull 若要作為 SOTA reconstruction 主結果，必須再 debug：
   - target scaling
   - RRCP_silver distribution
   - all-items / single-item model training
   - GraphLearner normalization

