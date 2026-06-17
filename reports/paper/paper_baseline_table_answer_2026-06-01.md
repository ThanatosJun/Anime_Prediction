# Baseline 主表問題回覆 2026-06-01

Status note, 2026-06-12: this is a historical baseline handoff document. It
records the earlier `n=2,808` complete-case discussion and the questions that
motivated the later sample-alignment rerun. Do not use this file as the current
Exp1 main-table source. Current results are documented under
`reports/experiments/sample_alignment/` and summarized in
`reports/planning/experiment_followup_todo_2026-06-12.md`.

本文件回覆文件整理 agent 提出的 23 個 baseline 表格問題。  
重要修正：最新版 `dev` 的 FusionModel v2 Run22 可在完整 test set `n=3,087` 上評估，因為它採用 missing-modality fallback；reference baseline 目前則採 strict complete-case intersection，因此多數 multimodal rows 是 `n=2,808`。這兩者不是同一種 evaluation policy，正式論文不能未加說明地混在同一張表比較。

## A. 樣本數與 Common Subset

### 1. 為什麼同時出現 `n=3,087` 和 `n=2,808`？

`n=3,087` 是 V2 test split 的完整樣本數，來源為：

```text
data/fussion/fusion_meta_clean_test_v2.csv
```

`n=2,808` 是目前 reference multimodal baseline 的 strict common subset：

```text
metadata ids ∩ project text embedding ids ∩ project image embedding ids
```

`F1-RF-Meta` 可以跑完整 `3,087`，因為它只需要 metadata。`F2/C1/C2/C3` 多數只剩 `2,808`，因為 reference baseline pipeline 會要求所需 artifact 全部存在才保留。

實際覆蓋狀況：

| artifact | test rows | metadata test ids 中缺少的筆數 |
|---|---:|---:|
| metadata | 3,087 | 0 |
| project text embedding | 2,808 | 279 |
| project image embedding | 3,087 | 0 |
| RAG none/selective/skapp_proxy/skapp_graph | 3,087 | 0 |
| GPT-2 text embedding | 3,087 | 0 |
| ResNet-50 image embedding | 3,087 | 0 |

所以被排除的 `279` 筆是缺 project text embedding，不是缺圖片、high-resolution image artifact 或 RAG。

但 FusionModel v2 Run22 不會排除這 279 筆。它保留 metadata full split 的所有 ids，並在 dataset `__getitem__` 中對缺失 embedding 補零/遮罩：

```python
text = self.text_map.get(anime_id, np.zeros(self.text_dim, dtype=np.float32))
image = self.image_map.get(
    anime_id,
    np.zeros((self.n_image_modality, self.image_dim), dtype=np.float32),
)
i_mask = self.image_mask_map.get(
    anime_id,
    np.ones(self.n_image_modality, dtype=bool),
)
```

因此 Run22 能跑 `3,087`，不是因為 text artifact 覆蓋完整，而是因為主框架支援 missing-modality fallback。

### 2. `2,808` subset 的產生規則是什麼？

對 reference multimodal rows 而言：

```text
metadata ids ∩ project text embedding ids ∩ project image embedding ids
```

對 RAG rows 而言，pipeline 還會額外要求 RAG feature parquet：

```text
metadata ids ∩ project text embedding ids ∩ project image embedding ids ∩ RAG feature ids
```

但目前 RAG feature 在 test split 覆蓋完整 `3,087`，所以沒有再減少樣本數。真正讓 `3,087` 變成 `2,808` 的原因是 project text embedding 只有 `2,808` 筆。

highres 與非 highres selected rows 的 test IDs 相同；highres 改的是 image embedding 的特徵值/維度，不是篩選規則。

### 3. `F2/C1/C2/C3` 的 `2,808` 是否保證是同一批 IDs？

是。已檢查以下 rows 在 `popularity` 與 `meanScore` 的 test IDs 完全一致：

- `F2-XGB-Concat`
- `C1-Armenta-ProjectInputProxy`
- `C2-ProjectInputCrossAttention`
- `C2-ProjectInputRecurrentFusion`
- `C3-RAG-Selective-XGB`
- `C3-ProjectInputSKAPPProxy-XGB`

這不是偶然，而是 `BaselineFeatureStore._resolve_ids()` 設計保證：它會根據 baseline 的 `feature_set` 對 metadata、embedding、RAG ids 做 intersection。

### 4. 能否補算 `F1-RF-Meta` 在 `2,808` common subset 上的 metrics？

可以，已補算。

| target | n_test | Spearman | log_MAE | log_R2 | factor_acc_2x | MAE | R2 | acc_within_10pt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 2,808 | 0.8507 | 0.8923 | 0.7602 | 0.4900 | 9334.9103 | 0.5798 | - |
| meanScore | 2,808 | 0.5634 | - | - | - | 8.0085 | 0.0791 | 0.6756 |

這個版本適合用於 reference baseline 彼此的 complete-case 比較。

### 5. 是否有完整 common-subset table？

有，已產出：

```text
reports/paper/paper_baseline_main_table_2026-06-01.csv
```

但它是 `n=2,808` complete-case reference baseline table。若要與 FusionModel v2 Run22 的 `n=3,087` full-test result 同表比較，這份還不是最終主表。

### 5b. 和 FusionModel v2 Run22 比較時要怎麼處理？

有兩種做法：

方案 A：補跑 full-test imputed baseline。

- 對 F2/C1/C2/C3 representative rows 採用 FusionModel v2 類似策略。
- 缺 project text embedding 的 279 筆補零向量。
- 若模型支援 mask，加入 missing mask；若不支援，需註明 zero-imputation。
- 產生 `n=3,087` baseline rows，才能和 Run22 直接放同一張主表。

方案 B：拆表。

- 主框架表：FusionModel v2 Run22，`n=3,087`。
- Reference baseline 表：complete-case reference baselines，`n=2,808`。
- 文字中明確說明 evaluation policy 不同，不做未加註解的單表排行。

若時間允許，建議方案 A；至少補 `F1-RF-Meta`、`F2-XGB-Concat`、`C2-ProjectInputRecurrentFusion`、`C3-RAG-Selective-XGB`、`C3-ProjectInputSKAPPProxy-XGB` 的 full-test imputed version。

## B. 主表 Baseline Selection Rule

### 6. 主表 baseline selection rule 是什麼？

主表不應把所有 baseline 塞進去，而應放能回答不同問題的代表 rows。  
但要先決定 evaluation policy：

- 若與 Run22 直接比較：主表 rows 應使用 `n=3,087` full-test imputed baseline。
- 若只比較 reference baselines：可使用目前 `n=2,808` complete-case table。

代表 rows 建議：

| baseline | 主表角色 | 選擇理由 |
|---|---|---|
| `F1-RF-Meta` | metadata-only strong baseline | 檢查多模態方法是否超越強 metadata-only baseline。 |
| `F2-XGB-Concat` | simple multimodal fusion | 檢查 deep/reference fusion 是否超越 simple concat。 |
| `C1-Armenta-ProjectInputProxy` | C1 代表 row | anime multimodal MLP literature-adapted reference。 |
| `C2-ProjectInputRecurrentFusion` | C2 primary row | 最完整保留 cross-modal + recurrent fusion 思想。 |
| `C2-ProjectInputCrossAttention` | C2 optional row | 隔離 cross-attention component，且 meanScore 較好。 |
| `C3-RAG-Selective-XGB` | selective retrieval reference | 代表 selective retrieval 策略。 |
| `C3-ProjectInputSKAPPProxy-XGB` | SKAPP-inspired performance row | 代表較穩定的 SKAPP-style project-input proxy，meanScore 最強。 |

### 7. `C1` 要選哪個版本？

主表建議選：

```text
C1-Armenta-ProjectInputProxy
```

理由：同 highres project-input、同 reference common-subset、最容易和 F2/C2/C3 公平比較。它應標為 project-input proxy / literature-adapted reference，不是 exact reproduction。

其他 C1 版本放 appendix 或開發紀錄：

- `C1-Armenta-ProjectInputReconstruction`
- `C1-Armenta-ProjectInputProxy-ResNet50`
- `C1-Armenta-Figure2Reconstruction`

### 8. `C2` 要選哪個版本？

Primary row：

```text
C2-ProjectInputRecurrentFusion
```

理由：比單純 cross-attention 更完整，保留 cross-modal attention + recurrent fusion。

Optional secondary row：

```text
C2-ProjectInputCrossAttention
```

理由：隔離 cross-attention component，且 meanScore 表現較好。

`C2-ProjectInputCTNNReconstruction` 與 `C2-ProjectInputCTNNDualVisualReconstruction` 建議放 appendix/future work，除非後續補出 full-test/highres 可比版本。

### 9. `C3` 要選哪個版本？

建議主表放兩個 C3 rows：

- `C3-RAG-Selective-XGB`：selective retrieval 策略代表。
- `C3-ProjectInputSKAPPProxy-XGB`：SKAPP-inspired retrieved aggregate proxy，也是目前 meanScore 最強 C3 row。

`C3-SourceExact-Staged-K64` 不放主表，只放 diagnostic paragraph 或 appendix diagnostic table。

### 10. 是否需要完整表放 appendix？

是。建議：

- 主文：代表 rows。
- Appendix：all baseline rows。
- Diagnostic appendix：`C3-SourceExact-Staged-K64`。

## C. C2 與原論文不一致問題

### 11. C2 沒有使用原論文模型，是否可接受？

可接受，但必須命名為：

- `C2-inspired`
- `C2-adapted`
- `C2 project-input proxy`
- `literature-adapted cross-modal fusion baseline`

避免：

- `C2 reproduction`
- `exact C2 reproduction`
- `original C2 model`

### 12. C2 哪些地方跟原論文不同？

保留：

- multimodal text-image fusion
- textual/visual cross-modal interaction
- `C2-ProjectInputRecurrentFusion` 的 recurrent/sequence-style fusion
- 統一 anime 任務上的 regression evaluation

未保留或不完全一致：

- 原論文是 movie box-office，本研究是 anime pre-release prediction。
- 原論文資料偏 movie reviews/posters，本研究用 AniList description、cover/banner、metadata。
- target、split、資料分布不同。
- encoder、fusion module、training environment 不是 source-exact。

### 13. C2 的 claim boundary 是什麼？

可以宣稱：

> We implement a C2-inspired, literature-adapted cross-modal fusion baseline on the same anime pre-release project inputs.

不可宣稱：

> We fully reproduce the original box-office revenue prediction model.

也不可宣稱這是原論文模型在 anime task 上的真實效果。

### 14. 如果 C2 沒有 exact reproduction，主文是否還應放 C2？

可以放，但必須標成 literature-adapted reference / project-input proxy。  
目前沒有可作主表的 exact C2 reproduction；更 source-faithful 的 CTNN / dual-visual 版本可放 appendix 或 future work。

## D. C3 與 Exp2 RAG 邊界

### 15. Exp1 的 C3 和 Exp2 的 RAG 消融怎麼切？

Exp1 的 C3 是 reference baseline family。  
Exp2 應保留給 proposed framework 的 RAG component ablation。

因此 C3 的 `none/sparse/dense/hybrid/selective` 不應直接寫成正式 Exp2，除非明確說那是 C3 reference-family ablation。

### 16. `C3-SourceExact-Staged-K64` 怎麼放？

只放 diagnostic paragraph 或 appendix diagnostic table，不放主表。

| target | diagnostic result |
|---|---|
| popularity | `log_R2=-2.1272`, `factor_acc_2x=0.0901`, raw `R2=-15.0432` |
| meanScore | `MAE=19.8518`, `acc_within_10pt=0.3061`, `R2=-4.2271` |

### 17. `C3-SourceExact-Staged-K64` 為什麼失敗？

目前不是普通誤差大，而是 prediction collapse / clipping boundary saturation。

可能原因：

- `top_k=64` 是 urgent reduced setting，不是 SKAPP 原始 `top_k=500`。
- target scaling / clipping 對 anime distribution 不穩。
- staged SKAPP/RRCP loss 與 prediction-space 假設不直接適配。
- SKAPP 原始任務與 anime pre-release prediction 分布不同。

## E. 指標與表格呈現

### 18. 主表要用哪些 metrics？

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

Selected rows 都可從 `test_predictions.csv` 重算。

### 19. 是否要保留 raw MAE / raw R2？

`popularity` raw MAE/raw R2 建議放 appendix/supporting table，主文以 log-space metrics + Spearman 為主。  
`meanScore` 則可直接使用 raw MAE/R2，因為它是 0 到 100 的線性尺度。

### 20. `log_MAE ≈ 0.89` 的直覺解釋是否合理？

合理，但要寫成近似直覺解釋：

> `log_MAE ≈ 0.89` 可直覺理解為約 `exp(0.89) ≈ 2.43x` 的幾何尺度偏差。

不要寫成每筆樣本都精確誤差 2.43 倍。

## F. 文件與可重現性

### 21. 哪些結果檔是 authoritative？

目前 reference complete-case table：

```text
reports/paper/paper_baseline_main_table_2026-06-01.csv
```

完整支援結果：

```text
reports/baselines/reference_baseline_metrics_extended_2026-06-01.csv
```

但若要和 Run22 full-test result 放同一張表，需新增：

```text
reports/paper_baseline_main_table_fulltest_imputed_2026-06-01.csv
```

或至少新增 `evaluation_policy` 欄位，區分：

- `complete_case_2808`
- `full_test_imputed_3087`

### 22. highres results 和 v2 results 怎麼區分？

reference multimodal rows 建議使用 highres results。  
highres 更新 image embedding artifact 的特徵值/維度，但沒有改變 selected rows 的 `2,808` IDs。

這次 `2,808` subset 不是 highres 造成，而是 project text embedding 覆蓋只有 `2,808`。FusionModel v2 Run22 能跑 `3,087` 是因為 zero fallback / mask。

### 23. 是否能產生正式的 `paper_baseline_main_table.csv`？

已產生 reference complete-case 版本：

```text
reports/paper/paper_baseline_main_table_2026-06-01.csv
```

但它應定位為 reference baseline complete-case table。若要與 Run22 full-test result 直接比較，仍需補 full-test imputed baseline table 或拆表呈現。

## 最重要的三點結論

1. `3,087` 是完整 V2 test split；`2,808` 是 reference multimodal complete-case subset。被排除的 `279` 筆是缺 project text embedding。FusionModel v2 Run22 能跑 `3,087` 是因為 missing-modality zero fallback / mask。

2. 主表若要和 Run22 直接比較，應補跑 `n=3,087` full-test imputed baseline；若時間不足，則拆成 proposed full-test table 與 reference complete-case table。

3. C2 不是 exact reproduction，正式文件應稱為 `C2-inspired` / `C2-adapted` / `C2 project-input proxy`，不可宣稱完整復現原論文。
