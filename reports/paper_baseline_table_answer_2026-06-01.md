# Baseline 主表問題回覆 2026-06-01

本文件回覆文件整理 agent 提出的 baseline 表格、樣本數、主表選擇、C2 claim boundary、C3 與 Exp2 邊界，以及正式論文應引用哪些結果檔等問題。

## A. 樣本數與 Common Subset

### 1. 為什麼同時出現 `n=3,087` 和 `n=2,808`？

`n=3,087` 是 V2 test split 的完整樣本數，來源為：

```text
data/fussion/fusion_meta_clean_test_v2.csv
```

`n=2,808` 是 multimodal baseline 的 strict common subset。其產生規則為：

```text
metadata ids ∩ project text embedding ids ∩ project image embedding ids
```

`F1-RF-Meta` 可以在完整 test set `3,087` 上跑，因為它只需要 metadata，不需要 text/image embedding。

`F2/C1/C2/C3` 多數只剩 `2,808`，因為它們使用 project text embedding 與 project image embedding；pipeline 會先對所需 artifact 的 id 做交集。

實際覆蓋狀況如下：

| artifact | test rows | metadata test ids 中缺少的筆數 |
|---|---:|---:|
| metadata | 3,087 | 0 |
| project text embedding | 2,808 | 279 |
| project image embedding | 3,087 | 0 |
| RAG none/selective/skapp_proxy/skapp_graph | 3,087 | 0 |
| GPT-2 text embedding | 3,087 | 0 |
| ResNet-50 image embedding | 3,087 | 0 |

所以被排除的 `279` 筆是因為缺 project text embedding，不是因為缺圖片、缺 high-resolution image artifact、缺 RAG feature、缺 GPT-2 artifact 或缺 ResNet-50 artifact。

### 2. `2,808` subset 的產生規則是什麼？

對使用 project text/image embedding 的 multimodal rows 而言，subset rule 是：

```text
metadata ids ∩ project text embedding ids ∩ project image embedding ids
```

對 RAG rows 而言，pipeline 會再額外要求 RAG feature parquet：

```text
metadata ids ∩ project text embedding ids ∩ project image embedding ids ∩ RAG feature ids
```

但目前 RAG feature 在 test split 有完整 `3,087` 覆蓋，因此 RAG 並沒有再減少樣本數。真正讓 test set 從 `3,087` 變成 `2,808` 的原因是 project text embedding 只有 `2,808` 筆。

highres 與非 highres selected rows 的 test IDs 相同。highres 更新的是 image embedding 的特徵值與維度，不是 common-subset 篩選規則。

### 3. `F2/C1/C2/C3` 的 `2,808` 是否保證是同一批 IDs？

是。已檢查以下 rows 在 `popularity` 與 `meanScore` 的 test IDs 完全一致：

- `F2-XGB-Concat`
- `C1-Armenta-ProjectInputProxy`
- `C2-ProjectInputCrossAttention`
- `C2-ProjectInputRecurrentFusion`
- `C3-RAG-Selective-XGB`
- `C3-ProjectInputSKAPPProxy-XGB`

這不是剛好一致，而是 pipeline 設計保證。`BaselineFeatureStore._resolve_ids()` 會根據該 baseline 的 `feature_set`，將 split ids 與所有必要的 embedding/RAG artifact ids 做 intersection，然後才建立 feature matrix。

### 4. 能否補算 `F1-RF-Meta` 在 `2,808` common subset 上的 metrics？

可以，已補算。

| target | n_test | Spearman | log_MAE | log_R2 | factor_acc_2x | MAE | R2 | acc_within_10pt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 2,808 | 0.8507 | 0.8923 | 0.7602 | 0.4900 | 9334.9103 | 0.5798 | - |
| meanScore | 2,808 | 0.5634 | - | - | - | 8.0085 | 0.0791 | 0.6756 |

正式主表應優先使用這個 `2,808` common-subset 版本，而不是直接拿 `F1 n=3,087` 跟 multimodal rows 的 `n=2,808` 比。

### 5. 是否有完整 common-subset table？

有，已產出：

```text
reports/paper_baseline_main_table_2026-06-01.csv
```

目前包含：

- `F1-RF-Meta`
- `F2-XGB-Concat`
- `C1-Armenta-ProjectInputProxy`
- `C2-ProjectInputCrossAttention`
- `C2-ProjectInputRecurrentFusion`
- `C3-RAG-Selective-XGB`
- `C3-ProjectInputSKAPPProxy-XGB`

並且同時包含 `popularity` 與 `meanScore`。

## B. 主表 Baseline Selection Rule

### 6. 主表 baseline selection rule 是什麼？

主表不建議把所有 baseline 全部塞進去。主表應放「能回答不同比較問題」的代表性 rows：

| baseline | 主表角色 | 選擇理由 |
|---|---|---|
| `F1-RF-Meta` | metadata-only strong baseline | 檢查多模態方法是否真的超越強 metadata-only baseline。 |
| `F2-XGB-Concat` | simple multimodal fusion | 檢查 deep/reference fusion 是否超越簡單 early concat。 |
| `C1-Armenta-ProjectInputProxy` | C1 代表 row | 代表 anime multimodal MLP literature-adapted reference。 |
| `C2-ProjectInputRecurrentFusion` | C2 primary row | 最完整保留 C2-inspired cross-modal + recurrent fusion 思想。 |
| `C2-ProjectInputCrossAttention` | C2 optional secondary row | 若表格空間允許，可用來隔離 cross-attention component；meanScore 表現也較好。 |
| `C3-RAG-Selective-XGB` | selective retrieval reference | 代表 selective retrieval 策略。 |
| `C3-ProjectInputSKAPPProxy-XGB` | SKAPP-inspired performance row | 代表目前較穩定的 SKAPP-style project-input proxy，meanScore 最強。 |

### 7. `C1` 要選哪個版本？

主表建議選：

```text
C1-Armenta-ProjectInputProxy
```

理由：

- 與 F2/C2/C3 selected rows 使用同一批 `2,808` common subset。
- 使用 highres project-input 設定，主表可比性最好。
- 它是「借用 C1 原論文 MLP 融合思想，映射到本專案輸入」的最乾淨版本。

其他 C1 版本建議放 appendix 或開發紀錄：

- `C1-Armenta-ProjectInputReconstruction`
- `C1-Armenta-ProjectInputProxy-ResNet50`
- `C1-Armenta-Figure2Reconstruction`

它們對完整度討論有價值，但不是主表中最公平、最易解釋的代表 row。

### 8. `C2` 要選哪個版本？

主表 primary row 建議選：

```text
C2-ProjectInputRecurrentFusion
```

理由是它比單純 cross-attention 更完整，保留了 C2-inspired 的 cross-modal attention 與 recurrent fusion 概念。

可選 secondary row：

```text
C2-ProjectInputCrossAttention
```

理由是它能隔離 cross-attention component，且在 meanScore 上表現較好。如果主表空間有限，只放 `C2-ProjectInputRecurrentFusion`；如果要呈現 C2 內部差異，可以兩個都放。

`C2-ProjectInputCTNNReconstruction` 與 `C2-ProjectInputCTNNDualVisualReconstruction` 建議放 appendix 或 future work，除非後續再產出高解析/common-subset 主表版本。

### 9. `C3` 要選哪個版本？

若主表空間允許，建議放兩個 C3 rows：

```text
C3-RAG-Selective-XGB
C3-ProjectInputSKAPPProxy-XGB
```

兩者角色不同：

- `C3-RAG-Selective-XGB`：代表 selective retrieval 策略。
- `C3-ProjectInputSKAPPProxy-XGB`：代表 SKAPP-inspired retrieved aggregate proxy，也是目前 meanScore 最強的 C3 row。

不建議把 `C3-SourceExact-Staged-K64` 放主表。它應放 diagnostic paragraph 或 appendix diagnostic table，因為兩個 target 都出現 boundary saturation。

### 10. 是否需要完整表放 appendix？

是，建議：

- 主文：只放 representative rows。
- Appendix：放 all baseline rows。
- Diagnostic appendix 或段落：放 `C3-SourceExact-Staged-K64`。

這樣主文不會被大量版本淹沒，也能保留完整實驗透明度。

## C. C2 與原論文不一致問題

### 11. C2 沒有使用原論文模型，是否可接受？

可以接受，但命名必須非常小心。正式文件應使用：

- `C2-inspired`
- `C2-adapted`
- `C2 project-input proxy`
- `literature-adapted cross-modal fusion baseline`

避免使用：

- `C2 reproduction`
- `exact C2 reproduction`
- `original C2 model`

### 12. C2 哪些地方跟原論文不同？

保留的核心思想：

- multimodal text-image fusion
- cross-modal interaction between textual and visual representations
- `C2-ProjectInputRecurrentFusion` 中保留 recurrent/sequence-style fusion 概念
- 在統一 anime 任務中做 regression evaluation

沒有保留或不完全一致的部分：

- 原論文是電影票房任務，本研究是 anime pre-release popularity/meanScore prediction。
- 原論文資料來源偏 movie reviews/posters，本研究使用 AniList description、cover/banner、metadata。
- 原論文 target、split、資料分布與本研究不同。
- 原論文 encoder、fusion module 與 training environment 未完全 source-exact 重現。
- 目前主表 C2 是 project-input adapted baseline，不是原始碼 exact reproduction。

### 13. C2 的 claim boundary 是什麼？

可以宣稱：

> We implement a C2-inspired, literature-adapted cross-modal fusion baseline on the same anime pre-release project inputs.

不可宣稱：

> We fully reproduce the original box-office revenue prediction model.

也不可宣稱：

> This is the original paper model's true performance on our anime task.

### 14. 如果 C2 沒有 exact reproduction，主文是否還應放 C2？

可以放，但必須標成 literature-adapted reference / project-input proxy。

目前沒有可作主表的 exact C2 reproduction。更 source-faithful 的 CTNN / dual-visual 版本可放 appendix 或 future work，除非後續再完成更嚴格的 highres/common-subset reproduction。

## D. C3 與 Exp2 RAG 邊界

### 15. Exp1 的 C3 和 Exp2 的 RAG 消融怎麼切？

Exp1 中的 C3 是 reference baseline family，用來跟 F1/F2/C1/C2 等 baseline 比較。

Exp2 應該保留給 proposed framework 的 RAG component ablation，例如：

- Proposed No-RAG
- Proposed metadata-only RAG
- Proposed text-only RAG
- Proposed hybrid RAG

因此目前 C3 的 `none/sparse/dense/hybrid/selective` 不應直接寫成正式 Exp2，除非明確說那是「C3 reference-family ablation」。在目前論文架構下，它們更適合放在 Exp1 baseline family 或 appendix analysis。

### 16. `C3-SourceExact-Staged-K64` 怎麼放？

`C3-SourceExact-Staged-K64` 已有 `popularity` 與 `meanScore`，但結果很差且出現 boundary saturation。

建議：

- 不放主表。
- 放 diagnostic paragraph。
- 可放 appendix diagnostic table。

目前結果：

| target | key diagnostic result |
|---|---|
| popularity | `log_R2=-2.1272`, `factor_acc_2x=0.0901`, raw `R2=-15.0432` |
| meanScore | `MAE=19.8518`, `acc_within_10pt=0.3061`, `R2=-4.2271` |

### 17. `C3-SourceExact-Staged-K64` 為什麼失敗？

目前看起來不是單純分數差，而是 prediction collapse / clipping boundary saturation。

可能原因：

- `top_k=64` 是 urgent reduced setting，不是 SKAPP 原始設定的 `top_k=500`。
- target scaling / clipping 對 anime target distribution 不穩。
- staged SKAPP/RRCP loss 與 prediction-space 假設不直接適配 anime popularity/meanScore。
- SKAPP 原始任務與資料分布和 anime pre-release prediction 差異很大。

正式文件可寫成：source-faithful SKAPP pipeline 的直接遷移目前不穩定，後續需要 top_k、target calibration、loss design 與 domain mapping 的重新調整。

## E. 指標與表格呈現

### 18. 主表要用哪些 metrics？

建議主表使用：

`popularity`：

- `Spearman_rho`
- `log_MAE`
- `log_R2`
- `factor_acc_2x`

`meanScore`：

- `Spearman_rho`
- `MAE`
- `R2`
- `acc_within_10pt`

這些 selected rows 都能從 `test_predictions.csv` 穩定重算。

### 19. 是否要保留 raw MAE / raw R2？

`popularity` 的 raw MAE / raw R2 建議放 appendix 或 supporting table。主文應以 log-space metrics 與 Spearman 為主，因為 popularity 是長尾分布，raw-scale 指標容易被少數極高人氣作品主導。

`meanScore` 則可直接使用 raw MAE / R2，因為它本身是 0 到 100 的線性尺度。

### 20. `log_MAE ≈ 0.89` 的直覺解釋是否合理？

合理，但要小心措辭。

可以寫：

> `log_MAE ≈ 0.89` corresponds to an approximate multiplicative error scale of `exp(0.89) ≈ 2.43x`.

中文可寫：

> `log_MAE ≈ 0.89` 可直覺理解為約 `exp(0.89) ≈ 2.43x` 的幾何尺度偏差。

但不要寫成「每一筆樣本都誤差 2.43 倍」，它只是 log-space 平均誤差的近似直覺解釋。

## F. 文件與可重現性

### 21. 哪些結果檔是 authoritative？

正式論文主表應引用：

```text
reports/paper_baseline_main_table_2026-06-01.csv
```

完整支援結果可引用：

```text
reports/reference_baseline_metrics_extended_2026-06-01.csv
```

其他檔案定位：

- `reports/reference_baseline_v2_results.csv`：V2 run-level source summary。
- `reports/reference_baseline_v2_highres_results.csv`：highres run-level source summary。
- `.exp/baseline/results/*/baseline_results.csv`：各 run 的原始輸出。

正式論文主表不建議從多個 run-level CSV 手工拼，應使用 `paper_baseline_main_table_2026-06-01.csv`，避免混到 full-test 與 common-subset rows。

### 22. highres results 和 v2 results 怎麼區分？

主表 multimodal rows 建議使用 highres results。

highres 主要更新 image embedding artifact，改變 image feature values / dimensions。它沒有改變 selected rows 的 `2,808` test IDs。

這次 `2,808` subset 的原因不是 highres，而是 project text embedding 只有 `2,808` 筆 test coverage。

### 23. 是否能產生正式的 `paper_baseline_main_table.csv`？

已產生：

```text
reports/paper_baseline_main_table_2026-06-01.csv
```

欄位包含：

- `baseline_id`
- `target`
- `common_subset`
- `subset_rule`
- `n_test`
- `role`
- `main_table_role`
- `reproduction_level`
- `claim_allowed`
- `Spearman_rho`
- `log_MAE`
- `log_R2`
- `factor_acc_2x`
- `MAE`
- `R2`
- `acc_within_10pt`
- `RMSE`
- `Pearson_r`
- `run_dir`

這份檔案應作為正式主表整理的入口。

## 最重要的三點結論

1. `3,087` 是完整 V2 test split；`2,808` 是 multimodal common subset。被排除的 `279` 筆是缺 project text embedding。`F1-RF-Meta` 已補算 `2,808` common-subset metrics。

2. 主表建議放代表性 rows：`F1-RF-Meta`、`F2-XGB-Concat`、`C1-Armenta-ProjectInputProxy`、`C2-ProjectInputRecurrentFusion`、可選 `C2-ProjectInputCrossAttention`、`C3-RAG-Selective-XGB`、`C3-ProjectInputSKAPPProxy-XGB`。完整表放 appendix。

3. C2 不是 exact reproduction，正式文件應稱為 `C2-inspired` / `C2-adapted` / `C2 project-input proxy`，不可宣稱完整復現原論文。
