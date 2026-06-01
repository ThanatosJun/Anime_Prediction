# 2026-05-19 會議用：EXP 與 Reference Baseline 對齊整理

## 一句話結論

目前完成度最高的是 **reference baseline / 文獻對照層**，不是正式
EXP1 / EXP2 ablation table 本體。

可在會議上說：

```text
Reference baselines 已大幅補齊。C1/C2 已可作為 EXP1 的 deep-learning literature baselines；
C3-RAG variants 已可作為 EXP2 的 RAG ablation 初版。

但 EXP1 的正式 controlled modality ablation 還沒有完全整理完成，
不能直接把 C1/C2/C3 全部說成 EXP1/EXP2 已完成。
```

## 目前應該整理成哪幾張表

這次會議建議整理三張表，不要混在一起：

| 表格 | 用途 | 放什麼 | 不該放什麼 |
|---|---|---|---|
| Reference baseline 結果對照 | 說明文獻 baseline 補到哪裡 | F0/F1/F2/T/I/C1/C2/C3 結果 | 不宣稱這就是 EXP1/EXP2 controlled ablation |
| EXP1 multimodal ablation progress | 說明模態貢獻實驗還缺什麼 | Metadata/Text/Image/RAG 組合是否已有正式 row | 不直接拿 C1/C2 當 ablation variant |
| EXP2 RAG ablation progress | 說明 RAG 實驗初版結果 | C3-RAG None/Sparse/Dense/Hybrid/Selective/SKAPP rows | 不把 SKAPPFull 性能差解讀成 RAG 無效 |

## EXP1 / EXP2 定義與目前對應

### EXP1：Modality Contribution

會議定義：

```text
EXP1：ablation study on multi-modal contribution
會議版本偏向：有 RAG 狀況下看 metadata/text/image 的貢獻
```

正式 EXP1 應該長這樣：

| EXP1 variant | Metadata | Text | Image | RAG | 目前是否已有正式 controlled row | 目前可暫時參考 |
|---|---|---|---|---|---|---|
| Metadata + RAG | yes | no | no | yes | 尚未乾淨完成 | 可由 C3 sparse/RAG feature 改出 |
| Metadata + Text + RAG | yes | yes | no | yes | 尚未乾淨完成 | 需新增 feature set |
| Metadata + Image + RAG | yes | no | yes | yes | 尚未乾淨完成 | 需新增 feature set |
| Metadata + Text + Image + RAG | yes | yes | yes | yes | partial | `C3-RAG-*` 可作初版 |

目前可支撐 EXP1 的 reference rows：

| Reference row | 對 EXP1 的角色 | 注意事項 |
|---|---|---|
| `F1-RF-Meta` / `F1-GB-Meta` | metadata-only classical reference | 不是 RAG 條件下的正式 EXP1 row |
| `T2-XGB-TextEmb` | text-only reference | 是單模態參照，不是有 RAG ablation |
| `I1-XGB-ImageEmb` | image-only reference | 是單模態參照，不是有 RAG ablation |
| `F2-XGB-Concat` | no-RAG full multimodal floor | 可當 EXP2 no-RAG baseline，也可輔助 EXP1 |
| `C1-Armenta-*` | anime multimodal DL literature baseline | 是文獻對照，不是 controlled ablation variant |
| `C2-ProjectInputCTNN*` | cross-modal transformer literature baseline | 是文獻對照，不是 controlled ablation variant |

目前 EXP1 完成度判斷：

```text
Reference 支撐足夠，但正式 controlled ablation table 尚未完成。
估計 EXP1 正式完成度：約 50-60%。
```

### EXP2：Retrieval Gain

會議定義：

```text
EXP2：ablation study on RAG / 有無 RAG / 不同 RAG 機制
```

目前 EXP2 已經比較接近可報告初版：

| EXP2 variant | Base input | RAG type | 目前對應 row | 狀態 |
|---|---|---|---|---|
| No-RAG | Metadata + Text + Image | none | `C3-RAG-None-XGB` 或 `F2-XGB-Concat` | 已有 |
| Metadata RAG | Metadata + Text + Image | sparse metadata retrieval | `C3-RAG-Sparse-XGB` | 已有 |
| Text RAG | Metadata + Text + Image | dense text retrieval | `C3-RAG-Dense-XGB` | 已有 |
| Hybrid RAG | Metadata + Text + Image | sparse + dense | `C3-RAG-Hybrid-XGB` | 已有 |
| Selective RAG | Metadata + Text + Image | contribution-filtered retrieval | `C3-RAG-Selective-XGB` | 已有 |
| SKAPP-style RAG | Metadata + Text + Image | learned/graph/full SKAPP proxy | `C3-ProjectInputSKAPP*` | 已有，但需解釋定位 |

目前 EXP2 完成度判斷：

```text
EXP2 基礎版已有初版結果。
但仍需把 C3 rows 重新整理成正式 EXP2 table，並明確指定 primary no-RAG baseline。
估計 EXP2 完成度：約 75-85%。
```

## Reference baseline 結果對照

### Common / Classical baselines

| baseline_id | target | test_MAE | test_R2 | Spearman | 用途 |
|---|---:|---:|---:|---:|---|
| `F0-Mean` | popularity | 15034.3970 | -0.1368 | 0.0000 | lowest floor |
| `F0-Mean` | meanScore | 10.4094 | -0.3536 | 0.0000 | lowest floor |
| `F0-Ridge-Meta` | popularity | 15222.9838 | -2.2072 | 0.7995 | linear metadata floor |
| `F0-Ridge-Meta` | meanScore | 8.5854 | 0.0075 | 0.5084 | linear metadata floor |
| `F1-RF-Meta` | popularity | 8590.0532 | 0.5811 | 0.8466 | strongest metadata classical row |
| `F1-RF-Meta` | meanScore | 7.9541 | 0.1298 | 0.5836 | strongest metadata classical row |
| `F1-GB-Meta` | popularity | 8917.8924 | 0.4951 | 0.8367 | metadata classical comparison |
| `F1-GB-Meta` | meanScore | 8.7243 | -0.0269 | 0.5380 | metadata classical comparison |

重點：

```text
Metadata-only classical baseline 很強，尤其 F1-RF-Meta。
這代表後續 multimodal/RAG 必須超過 metadata floor 才有說服力。
```

### No-RAG / single-modality references

| baseline_id | target | test_MAE | test_R2 | Spearman | 用途 |
|---|---:|---:|---:|---:|---|
| `F2-XGB-Concat` | popularity | 9588.2590 | 0.5194 | 0.8575 | no-RAG multimodal floor |
| `F2-XGB-Concat` | meanScore | 8.3391 | 0.0193 | 0.5292 | no-RAG multimodal floor |
| `T2-XGB-TextEmb` | popularity | 14908.8897 | -0.0152 | 0.6488 | text-only reference |
| `T2-XGB-TextEmb` | meanScore | 10.3206 | -0.3846 | 0.2427 | text-only reference |
| `I1-XGB-ImageEmb` | popularity | 13815.0865 | 0.0158 | 0.6046 | image-only reference |
| `I1-XGB-ImageEmb` | meanScore | 9.4042 | -0.1559 | 0.2918 | image-only reference |

重點：

```text
Text/image 單獨不強，但有排序訊號。
No-RAG multimodal `F2-XGB-Concat` 是 EXP2 中很適合當 primary no-RAG baseline 的 row。
```

### C1 / C2 deep-learning literature baselines

| baseline_id | target | test_MAE | test_R2 | Spearman | 定位 |
|---|---:|---:|---:|---:|---|
| `C1-Armenta-ProjectInputReconstruction` | popularity | 10719.7513 | 0.2898 | 0.8192 | C1 project-input 主線 |
| `C1-Armenta-ProjectInputReconstruction` | meanScore | 9.0250 | -0.1096 | 0.4666 | C1 project-input 主線 |
| `C1-Armenta-Figure2Reconstruction` | popularity | 11878.6328 | 0.3556 | 0.7823 | C1 Figure 2 旁支 |
| `C1-Armenta-Figure2Reconstruction` | meanScore | 9.7747 | -0.2172 | 0.3824 | C1 Figure 2 旁支 |
| `C2-ProjectInputCTNNReconstruction` | popularity | 10151.2161 | 0.4608 | 0.8471 | C2 主線 |
| `C2-ProjectInputCTNNReconstruction` | meanScore | 8.1751 | 0.0696 | 0.5247 | C2 主線 |
| `C2-ProjectInputCTNNDualVisualReconstruction` | popularity | 10214.6356 | 0.4421 | 0.8491 | C2 dual-visual diagnostic |
| `C2-ProjectInputCTNNDualVisualReconstruction` | meanScore | 8.8957 | -0.0720 | 0.5310 | C2 dual-visual diagnostic |

重點：

```text
C1/C2 是 EXP1 的 deep-learning literature baselines，不是 EXP1 ablation variants。
C2 比 C1 更接近可用；C2 dual-visual 改善 Spearman/log_MAE，但 R2 不如 C2 主線。
```

### C3 / EXP2 RAG rows

| baseline_id | target | test_MAE | test_R2 | Spearman | 對應 EXP2 |
|---|---:|---:|---:|---:|---|
| `C3-RAG-None-XGB` | popularity | 9664.2004 | 0.5064 | 0.8583 | no-RAG control |
| `C3-RAG-None-XGB` | meanScore | 8.3647 | 0.0132 | 0.5307 | no-RAG control |
| `C3-RAG-Sparse-XGB` | popularity | 9736.1037 | 0.5725 | 0.8722 | metadata/sparse RAG |
| `C3-RAG-Sparse-XGB` | meanScore | 8.1703 | 0.0730 | 0.5384 | metadata/sparse RAG |
| `C3-RAG-Dense-XGB` | popularity | 9704.8621 | 0.5084 | 0.8584 | text/dense RAG |
| `C3-RAG-Dense-XGB` | meanScore | 8.2445 | 0.0464 | 0.5382 | text/dense RAG |
| `C3-RAG-Hybrid-XGB` | popularity | 10327.0456 | 0.4828 | 0.8537 | hybrid RAG |
| `C3-RAG-Hybrid-XGB` | meanScore | 8.3798 | 0.0307 | 0.5539 | hybrid RAG |
| `C3-RAG-Selective-XGB` | popularity | 9782.2338 | 0.5775 | 0.8746 | selective RAG |
| `C3-RAG-Selective-XGB` | meanScore | 8.0914 | 0.0905 | 0.5470 | selective RAG |
| `C3-ProjectInputSKAPPProxy-XGB` | popularity | 10239.2909 | 0.5170 | 0.8574 | SKAPP aggregate proxy |
| `C3-ProjectInputSKAPPProxy-XGB` | meanScore | 8.1715 | 0.0744 | 0.5369 | SKAPP aggregate proxy |
| `C3-ProjectInputSKAPPGraphProxy` | popularity | 11501.8681 | 0.4404 | 0.8561 | SKAPP architecture proxy |
| `C3-ProjectInputSKAPPGraphProxy` | meanScore | 8.1448 | 0.0690 | 0.4973 | SKAPP architecture proxy |
| `C3-ProjectInputSKAPPFull` | popularity | 14668.1228 | -0.4927 | 0.6985 | SKAPP full reconstruction diagnostic |
| `C3-ProjectInputSKAPPFull` | meanScore | 9.8063 | -0.2385 | 0.3657 | SKAPP full reconstruction diagnostic |

重點：

```text
目前 EXP2 最強 performance row 是 `C3-RAG-Selective-XGB`。
Sparse RAG 已明顯超過 no-RAG；Selective RAG 又略優於 Sparse。
SKAPPFull 已跑通完整構造，但 performance 很差，目前只能當 reconstruction diagnostic。
```

## 這次會議建議報告口徑

### 可以說

```text
1. Reference baseline 已補齊到可報告狀態：F0/F1/F2/T/I/C1/C2/C3 都已有結果。
2. C1/C2 已不再停留在 Lite/proxy，已完成 project-input reconstruction。
3. C3-RAG none/sparse/dense/hybrid/selective 已可整理成 EXP2 基礎版。
4. 目前最有正向效果的是 RAG selective/sparse；C3-RAG-Selective-XGB 在 popularity R2/Spearman 最佳。
5. EXP1 仍需要整理正式 controlled ablation，不能直接用 C1/C2 取代。
```

### 不建議說

```text
1. 不要說 EXP1 已完成。
2. 不要說 C1/C2/C3 是 exact reproduction。
3. 不要說 SKAPPFull performance 代表 RAG 無效；它只是 full reconstruction 尚未調好。
4. 不要把 C1 Figure2 旁支當成本專案主框架 baseline。
5. 不要把 C2 dual-visual diagnostic 當 C2 主線替代品。
```

## 目前與會議目標的差距

| 項目 | 會議期待 | 目前狀態 | 差距 |
|---|---|---|---|
| Reference baseline | C1/C2/C3 基礎復現與分數 | 已大幅補齊 | 約 10-15%：剩 claim/tuning |
| EXP1 | 有 RAG 的 modality ablation 基礎版 | 只有 reference 支撐，正式表未完成 | 約 40-50% |
| EXP2 | 有無 RAG / RAG 類型 ablation 基礎版 | C3-RAG variants 已有 | 約 15-25%：需整理成正式 EXP2 表 |
| EXP3 | final robustness/generalization 設計 | 尚未定案 | 約 60-70% |
| 主 fusion model | 整合組員 embedding/image/RAG 改動 | reference 端清楚，主模型需整合 | 約 30% |

## 下一步工作拆分

### 會議後短期要做

1. 正式建立 EXP1 feature sets：
   - Metadata + RAG
   - Metadata + Text + RAG
   - Metadata + Image + RAG
   - Metadata + Text + Image + RAG
2. 正式整理 EXP2 table：
   - No-RAG
   - Sparse / metadata RAG
   - Dense / text RAG
   - Hybrid RAG
   - Selective RAG
   - SKAPP-style diagnostic
3. 決定 EXP3：
   - time-period split / robustness
   - 或 AniList vs MAL out-dataset generalization
4. 將組員進度接回主實驗：
   - e5 embedding
   - YOLO character image embedding
   - metadata TE / feature filtering
   - RAG enable-disable 與降噪

### 建議會議決議

```text
1. EXP1 是否確認採「有 RAG 條件下的 modality ablation」？
2. EXP2 的 no-RAG baseline 要用 F2-XGB-Concat 還是 C3-RAG-None-XGB？
3. EXP3 最終採 time split robustness 還是 AniList/MAL out-dataset？
4. Reference baseline 表是否獨立放一節，不混進 EXP1/EXP2 ablation 表？
```

