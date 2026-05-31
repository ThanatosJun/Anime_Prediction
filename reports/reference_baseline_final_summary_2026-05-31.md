# Reference Baseline Final Summary 2026-05-31

本文整理 C1 / C2 / C3 reference baselines 的論文定位、目前可用結果、high-resolution embedding 重跑後的變化，以及寫成果論文時應該主動交代的限制。

## 1. 定位與分類

先定義本文使用的分類：

- `External-adapted`：可以作為外部文獻 baseline 放進主論文，但只能宣稱「依文獻架構改寫到本專案輸入」，不可稱為 exact reproduction。
- `Project-input proxy`：以本專案 metadata / text embedding / image embedding 為輸入，借用文獻部分結構或概念；適合作為內部對照或附錄，不適合獨立宣稱為完整外部 baseline。
- `Development / appendix only`：曾經有助於開發或診斷，但不建議佔用主論文主表。

| Route | 對應論文 | 目前代表 row | 分類 | 主論文建議 | 判斷理由 |
|---|---|---|---|---|---|
| C1 | Armenta-Segura & Sidorov 2025, *Anime popularity prediction before huge investments: a multimodal approach using deep learning* | `C1-Armenta-ProjectInputReconstruction` | External-adapted | 可放主表，但需加註 adapted | V2 已補 GPT-2 synopsis、ResNet-50 cover/banner、Armenta-shaped branch + Big MLP；但仍以 project metadata / cover / banner 取代原論文 main-character descriptions / portraits，因此不是 exact reproduction。 |
| C1 | 同上 | `C1-Armenta-ProjectInputProxy` | Project-input proxy | 不建議主表；可放附錄或 development record | high-res 已跑完，但它使用 project text/image embeddings，不是 GPT-2 + ResNet-50，也不是 Figure 2 的 character branch；適合回答「原論文 fusion shape 在本專案輸入上是否有效」。 |
| C1 | 同上 | `C1-Armenta-Figure2Reconstruction` | Side reconstruction / pending | 不建議主表 | 這條最貼近 Figure 2 character-centric setup，但目前不是本專案主輸入 contract，且 character artifact 覆蓋率與任務對標都會被質疑。 |
| C2 | Madongo, Tang & Hassan 2023, *Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers* | `C2-ProjectInputCTNNReconstruction` | External-adapted | 可放主表，但需加註 adapted | V2 已補 transformer encoders、bidirectional cross attention、GRU recurrent fusion、metadata factor gate；但資料集、任務、poster/review 原始抽特徵方式不同。 |
| C2 | 同上 | `C2-ProjectInputCrossAttention` | Project-input proxy | 附錄或 ablation-like proxy | high-res 已跑，能說明 cross-attention 對本專案輸入的效果，但未完整包含原 CTNN recurrent / source feature extraction contract。 |
| C2 | 同上 | `C2-ProjectInputRecurrentFusion` | Project-input proxy | 附錄或 ablation-like proxy | high-res 已跑，補了 recurrent fusion 概念，performance 不錯，但仍不是 source-aligned reconstruction。 |
| C2 | 同上 | `C2-CTNN-Lite` | Development / appendix only | 不建議主表 | 太簡化，只有 text-image two-token transformer；可以當早期 milestone，不應代表 C2 外部 baseline。 |
| C3 | Xu et al. 2025 SKAPP, *Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation* | `C3-RAG-Selective-XGB` | External-inspired performance baseline | 可放主表 | 它不是 SKAPP exact reproduction，但 selective retrieval 確實帶來最強 popularity R2；適合當「SKAPP-inspired selective retrieval baseline」。 |
| C3 | 同上 | `C3-ProjectInputSKAPPProxy-XGB` | Project-input proxy | 可放主表或附錄，取決於篇幅 | learned contribution + attention-weighted aggregate 對 meanScore 最有效；架構不如 GraphProxy 接近 SKAPP，但 performance 具有價值。 |
| C3 | 同上 | `C3-ProjectInputSKAPPGraphProxy` | Architecture proxy | 建議附錄；若主表需清楚標成 architecture proxy | 有 retrieved tensor、RRCP-style mask、learned graph adjacency、contribution-aware attention；但特徵極寬且 popularity MAE/R2 弱，不適合當 performance 主張。 |
| C3 | 同上 | `C3-RAG-None/Sparse/Dense/Hybrid-XGB` | Internal retrieval comparison | 附錄或 RAG section 小表 | Sparse/Selective 有明顯幫助；None/Dense/Hybrid 主要是 retrieval strategy 對照，不應包裝成完整外部 baseline。 |

### 主論文可用名單

若主表只放最精簡版本：

| 主表角色 | 建議 row | 使用原因 |
|---|---|---|
| C1 external-adapted | `C1-Armenta-ProjectInputReconstruction` | 最接近 C1 論文的 GPT-2 + ResNet-50 + Big MLP route。 |
| C2 external-adapted | `C2-ProjectInputCTNNReconstruction` | 最接近 C2 論文的 transformer encoder + cross attention + recurrent fusion route。 |
| C3 retrieval performance | `C3-RAG-Selective-XGB` | high-res 後 popularity R2 最強。 |
| C3 SKAPP-style score performance | `C3-ProjectInputSKAPPProxy-XGB` | high-res 後 meanScore MAE/R2 最強。 |

若主表篇幅有限，`C3-ProjectInputSKAPPGraphProxy` 建議放附錄，因為它的架構對齊價值高，但 performance 和計算成本都不利於主張。

## 2. 目前最佳結果

以下使用 `reports/reference_baseline_v2_highres_results.csv`。注意：high-res rows 多數使用 project text/image embedding 的 available-case subset，`n_test = 2808`；image-only 是 `n_test = 3087`。正式論文若要嚴格比較，應另產同一批 test ids 的 common-subset table。

### Popularity 最佳結果

| 排名依據 | baseline_id | test_MAE | test_R2 | test_Spearman_rho | 解讀 |
|---|---|---:|---:|---:|---|
| 最佳 MAE | `C3-RAG-Sparse-XGB` | 9219.1995 | 0.6141 | 0.8715 | sparse metadata retrieval 對絕對誤差最有效。 |
| 最佳 R2 | `C3-RAG-Selective-XGB` | 9256.1195 | 0.6182 | 0.8719 | selective filtering 對解釋變異最佳，是目前 C3 performance 主力。 |
| 最佳 Spearman | `C3-ProjectInputSKAPPGraphProxy` | 11254.5741 | 0.3862 | 0.8737 | 排序能力很好，但絕對值校準差；適合討論 ranking vs regression。 |

### meanScore 最佳結果

| 排名依據 | baseline_id | test_MAE | test_R2 | test_Spearman_rho | 解讀 |
|---|---|---:|---:|---:|---|
| 最佳 MAE / R2 | `C3-ProjectInputSKAPPProxy-XGB` | 7.8582 | 0.1274 | 0.5634 | learned contribution + attention-weighted retrieved aggregate 對 score 最有效。 |
| 次佳 R2 | `C3-RAG-Sparse-XGB` | 8.0691 | 0.0938 | 0.5563 | sparse retrieval 仍穩定有效。 |
| 最佳 Spearman | `C3-RAG-Hybrid-XGB` | 8.1344 | 0.0724 | 0.5657 | hybrid retrieval 對排序略有幫助，但 MAE/R2 不如 sparse/selective/proxy。 |

## 3. High-res embedding 相比 V2 的改善

high-res 新 image parquet 將 image feature 從 1024 維提升到 3075 維，multimodal concat 從 1559 維提升到 3994 維，C3 graph proxy 從 15695 維提升到 42480 維。

### 最大亮點

| baseline_id | target | V2 R2 | high-res R2 | delta R2 | delta MAE | delta Spearman | 判讀 |
|---|---|---:|---:|---:|---:|---:|---|
| `I1-XGB-ImageEmb` | popularity | -0.0039 | 0.2096 | +0.2135 | -1827.3454 | +0.1437 | 高解析 image embedding 對 image-only 幫助非常大，舊圖像特徵確實是瓶頸。 |
| `I1-XGB-ImageEmb` | meanScore | -0.1603 | -0.0103 | +0.1500 | -0.7816 | +0.1354 | score 仍接近不可解釋，但已從明顯負 R2 拉近到接近 0。 |
| `F2-XGB-Concat` | popularity | 0.5108 | 0.5515 | +0.0407 | -148.5959 | +0.0071 | multimodal concat 有受益，但幅度遠小於 image-only。 |
| `F2-XGB-Concat` | meanScore | -0.0231 | 0.0562 | +0.0793 | -0.3442 | +0.0428 | high-res 讓 F2 的 score 從負 R2 變成正 R2。 |
| `C3-RAG-Selective-XGB` | popularity | 0.5901 | 0.6182 | +0.0281 | -264.1027 | +0.0000 | selective retrieval 已經很強，high-res 主要改善誤差與 R2，不改變排序。 |
| `C3-ProjectInputSKAPPProxy-XGB` | meanScore | 0.0472 | 0.1274 | +0.0802 | -0.4048 | +0.0417 | SKAPP aggregate proxy 是 high-res 後 meanScore 最大受益者之一。 |

### 反常點

| baseline_id | target | 現象 | 可能原因 |
|---|---|---|---|
| `C3-ProjectInputSKAPPGraphProxy` | popularity | MAE 改善 -257.4336、Spearman +0.0174，但 R2 下降 -0.0184 | 排序變好但絕對值校準或 outlier variance 變差；GraphProxy 很寬且模型容量高，可能對 popularity scale 不穩。 |
| `C2-ProjectInputCrossAttention` | popularity | R2 +0.0313、Spearman +0.0174，但 MAE 反而增加 +149.1253 | 高解析特徵可能改善整體變異解釋，卻在部分樣本絕對誤差變大；需要看 residual by popularity bucket。 |
| `C2-CTNN-Lite` | meanScore | R2 +0.2150 是最大增幅，但 high-res 後仍為 -0.0356 | 改善幅度大不代表結果可用；它從很差變成沒那麼差，仍不適合作 C2 主線。 |

## 4. C3 RAG 類 baseline 哪些真的有效

以下同時以 `C3-RAG-None-XGB` 和 `F2-XGB-Concat` 作比較。`C3-RAG-None-XGB` 是 RAG schema 的 no-retrieval control，`F2-XGB-Concat` 是純 metadata + text + image concat baseline。

### Popularity

| baseline_id | vs None delta MAE | vs None delta R2 | vs F2 delta MAE | vs F2 delta R2 | 判斷 |
|---|---:|---:|---:|---:|---|
| `C3-RAG-Sparse-XGB` | -362.9108 | +0.0701 | -320.5052 | +0.0626 | 真的有效；而且 MAE 最佳。 |
| `C3-RAG-Selective-XGB` | -325.9908 | +0.0742 | -283.5852 | +0.0667 | 真的有效；R2 最佳，可作 C3 主力。 |
| `C3-RAG-Dense-XGB` | +80.0195 | +0.0062 | +122.4251 | -0.0013 | 幾乎沒有實質提升，且 MAE 變差；可作對照組。 |
| `C3-RAG-Hybrid-XGB` | +385.5417 | -0.0128 | +427.9473 | -0.0203 | 徒增複雜度；目前不支持 hybrid retrieval。 |
| `C3-ProjectInputSKAPPProxy-XGB` | +493.1440 | -0.0010 | +535.5496 | -0.0085 | 對 popularity 無效；它的價值主要在 meanScore。 |
| `C3-ProjectInputSKAPPGraphProxy` | +1672.4638 | -0.1578 | +1714.8694 | -0.1653 | performance 不適合作主張；只保留架構診斷價值。 |

### meanScore

| baseline_id | vs None delta MAE | vs None delta R2 | vs F2 delta MAE | vs F2 delta R2 | 判斷 |
|---|---:|---:|---:|---:|---|
| `C3-ProjectInputSKAPPProxy-XGB` | -0.3537 | +0.0691 | -0.3449 | +0.0712 | 真的有效；目前 meanScore 最佳。 |
| `C3-RAG-Sparse-XGB` | -0.1428 | +0.0355 | -0.1340 | +0.0376 | 有效且穩定。 |
| `C3-RAG-Selective-XGB` | -0.1218 | +0.0301 | -0.1130 | +0.0322 | 有效，但 meanScore 不如 SKAPPProxy / Sparse。 |
| `C3-RAG-Dense-XGB` | -0.1182 | +0.0223 | -0.1094 | +0.0244 | 小幅有效，可放附錄。 |
| `C3-RAG-Hybrid-XGB` | -0.0775 | +0.0141 | -0.0687 | +0.0162 | 提升有限；較像對照組。 |
| `C3-ProjectInputSKAPPGraphProxy` | -0.0901 | +0.0088 | -0.0813 | +0.0109 | 小幅改善但 Spearman 下降，複雜度不划算。 |

結論：C3 的有效訊號不是「所有 RAG 都有效」，而是「sparse metadata retrieval 與 selective filtering 對 popularity 有效；learned contribution aggregate 對 meanScore 有效」。Dense / Hybrid / GraphProxy 目前都比較像必要對照，而非主力。

## 5. 隱藏精華與盲點

### 5.1 Metadata 可能比我們想像中更強

V2 的 `F1-RF-Meta` 在 metadata-only 設定下已達 popularity `test_R2 = 0.5865`、`test_MAE = 8551.7168`。這表示播出年份、類型、前作、studio、voice actor 等 pre-release metadata 已經吃掉大量 popularity 訊號。

這對論文很重要：不能只說「multimodal 一定比 metadata 好」。更合理的寫法是：

- metadata 是強 baseline；
- text/image 單模態不穩；
- RAG 對 popularity 的增益主要體現在 metadata 強 baseline 之上的排序/相似歷史樣本補充；
- high-res image embedding 改善 image-only，但還需要更好的 fusion 才能把影像訊號完整轉成 final gain。

### 5.2 不同 rows 的 test set 大小不完全一致

目前結果表中存在不同 `n_test`：

- metadata-only / image-only / reconstruction rows 常見 `n_test = 3087`
- project text + image multimodal / C3 rows 常見 `n_test = 2808`

這代表目前表格是 available-case evaluation，不是完全相同 test ids 的 closed comparison。正式主論文若要嚴格比較，需要另產 common-subset result，至少對主表 rows 使用同一批 test ids。否則 `F1-RF-Meta`、`F2-XGB-Concat`、C1/C2/C3 的 MAE 不能被過度直接比較。

### 5.3 High-res image embedding 的主要價值被 fusion 稀釋

`I1-XGB-ImageEmb` popularity R2 從 -0.0039 到 0.2096，是非常明確的改善；但 `F2-XGB-Concat` popularity 只從 0.5108 到 0.5515。這可能代表：

- high-res 圖像特徵確實變好；
- 但 concat + XGB 對 3075 維 image feature 的利用有限；
- metadata/text 可能在 fusion 中主導；
- 後續主框架若有 gate / projection / attention，應該能把這點寫成方法動機。

### 5.4 GraphProxy 排序強、回歸弱

`C3-ProjectInputSKAPPGraphProxy` 在 popularity 的 Spearman 是 0.8737，甚至高於 Sparse/Selective；但 MAE = 11254.5741、R2 = 0.3862 明顯落後。

這很像「模型知道誰比誰熱門，但不知道熱門程度要差多少」。這可以發展成 EXP3 或 error analysis：

- popularity bucket residual；
- top-k ranking quality；
- high popularity outlier calibration；
- log target inverse-transform 後的 extreme error。

### 5.5 Sparse retrieval 比 dense/hybrid 更可靠

Popularity 上 sparse / selective 明顯贏 dense / hybrid。這暗示對 anime popularity 來說，genre、studio、source、voice actor 等結構化相似度，比純 text embedding semantic similarity 更有效。這個發現很適合寫進 RAG 設計動機，也能支持 metadata-aware retrieval 的必要性。

### 5.6 meanScore 的可預測性仍低

即使 high-res 後，meanScore 最佳 R2 也只有 0.1274。這不是單一 baseline 的問題，而是跨 C1/C2/C3/F2 都偏低。論文中應誠實交代：

- popularity 較能被 pre-release metadata / history / retrieval 解釋；
- meanScore 可能更依賴播出後品質、觀眾口碑、製作完成度，pre-release 訊號較弱；
- meanScore 應避免被包裝成與 popularity 同樣可預測。

## 6. 程式與參數層面的限制

| 限制 | 位置 | 影響 |
|---|---|---|
| C1 ProjectInput MLP 是 synopsis branch + project context branch，不是原 Figure 2 character branch | `src/reference_baseline_branch/sklearn_models.py:1212` | 可宣稱 Armenta-style project-input reconstruction/proxy，不可宣稱 exact Figure 2。 |
| C2 CTNN reconstruction 有 text/image transformer encoder、cross attention、metadata gate、GRU recurrent fusion | `src/reference_baseline_branch/sklearn_models.py:1562` | 這是 C2 最可 defend 的 external-adapted row；但仍不是 movie poster/review 原始任務。 |
| C3 GraphProxy 使用 fixed top-k retrieved tensor、RRCP-style mask、learned adjacency、attention | `src/reference_baseline_branch/sklearn_models.py:1900` | 架構接近度高，但仍是 proxy；且 42480 features 計算成本很高。 |
| SKAPP selective 目前用 median threshold 的 deterministic/proxy selection | `src/reference_baseline_branch/build_c3_rag_features.py:221` | 不可宣稱完整 RRCP；只能說 RRCP-style contribution filtering。 |
| Dense retrieval 只用 project text embeddings 與 train matrix 相似度 | `src/reference_baseline_branch/build_c3_rag_features.py:330` | 若 text embedding 本身不適合 popularity retrieval，dense/hybrid 會帶噪音。 |
| GraphProxy artifact 需要 chunked parquet 寫出 | `src/reference_baseline_branch/build_c3_rag_features.py:101` | 表示此路線計算與儲存成本高，不適合當簡潔主線 baseline。 |
| runner 會依 feature names 動態改寫 model dims | `src/reference_baseline_branch/run_reference_baselines.py:180` | config 中舊維度數字不一定等於實際輸入維度；論文應以 result table 的 `n_features` 為準。 |

## 7. 建議寫進週會報告的版本

### 進度

- 已完成 C1 / C2 / C3 的 reference baseline family 整理。
- 已完成 V2 與 high-resolution image embedding rerun 結果表。
- high-res 後，image-only 明顯改善；C3 Sparse / Selective 是 popularity 最強；C3 SKAPPProxy 是 meanScore 最強。

### 難點

- C1/C2/C3 都不是 exact reproduction，因為原論文資料集、輸入欄位、任務定義與本專案不同。
- 目前部分 rows 的 test ids 不完全一致，正式主表需要 common-subset evaluation。
- C3 GraphProxy 架構最複雜，但 performance 不如簡單 sparse/selective XGB，需作為 limitation 或 appendix。

### 預期進度

1. 產出 common-subset 主表，避免不同 n_test 造成比較爭議。
2. 決定主論文 C1/C2 使用 V2 reconstruction，還是補 high-res-compatible reconstruction 註記。
3. 對 C3 做 residual / bucket analysis，確認 GraphProxy 高 Spearman 低 R2 的原因。
4. 將主框架 EXP1 / EXP2 與 reference baseline 分表呈現，避免把 external baseline 誤當成 ablation experiment。
