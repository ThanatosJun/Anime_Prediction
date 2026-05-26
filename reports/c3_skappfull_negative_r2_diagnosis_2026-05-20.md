# C3 SKAPPFull 負 R2 診斷 - 2026-05-20

本文記錄 `C3-ProjectInputSKAPPFull` run 35 為何出現負 R2。

結論先寫在前面：目前負 R2 不是因為 tensor 檔案壞掉、mask 全空、或 target 欄位讀反。主要原因分成兩個：

1. `popularity` 在 log 空間其實仍有正向解釋力，但轉回原始人氣尺度後產生少數極端高估值，RMSE 被放大，導致原尺度 R2 變成負值。
2. `meanScore` 則是 final model 明顯 underfit，且 test split 的分布比 train 高；模型輸出平均值仍貼近 train mean，因此 test R2 變負。

## 檢查對象

```text
.exp/baseline/results/35
.exp/baseline/skapp_full/dataset/train.npz
.exp/baseline/skapp_full/dataset/val.npz
.exp/baseline/skapp_full/dataset/test.npz
src/reference_baseline_branch/run_c3_skapp_full.py
baseline_refer/skapp-main/src/
```

run 35 結果：

| target | val_MAE | val_R2 | val_Spearman | test_MAE | test_R2 | test_Spearman | test_log_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 16327.5980 | -2.7417 | 0.7411 | 14668.1228 | -0.4927 | 0.6985 | 1.2983 |
| meanScore | 8.5718 | 0.1626 | 0.4614 | 9.8063 | -0.2385 | 0.3657 |  |

## 已排除的問題

### 1. Dataset tensor 並沒有明顯壞掉

三個 split 的 embedding coverage 正常：

| split | query_text nonzero | query_image nonzero | retrieved_text nonzero | retrieved_image nonzero | mask mean | empty rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.9606 | 1.0000 | 0.9955 | 0.9959 | 0.9959 | 39 |
| val | 0.9037 | 1.0000 | 0.9990 | 1.0000 | 1.0000 | 0 |
| test | 0.9096 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

這代表 C3Full 的主要輸入不是空資料。少數 query text 為零，但 image 與 retrieved features 幾乎都有值，不足以單獨解釋負 R2。

### 2. target transform 方向沒有顛倒

`run_c3_skapp_full.py` 的設定：

```text
popularity: log1p target, prediction 後 expm1 回原始尺度
meanScore: identity target
```

retrieved label 也與此一致：

```text
retrieved popularity label: log1p(popularity)
retrieved meanScore label: meanScore / 100，載入 target 時再乘回 100
```

所以目前不是 target 單位讀錯或 label index 對錯造成的。

## popularity 為何變負

`popularity` 的原尺度 R2 是負的，但 log 空間不是負的：

| split | raw_R2 | log_R2 | raw_MAE | log_MAE |
|---|---:|---:|---:|---:|
| val | -2.7417 | 0.5520 | 16327.60 | 1.1989 |
| test | -0.4927 | 0.4727 | 14668.12 | 1.2983 |

這表示模型仍有排序/粗略解釋能力，所以 Spearman 才還有 `0.6985`。問題是 log 空間預測一旦高估，經過 `expm1` 回到原始 popularity 後會爆成極端 outlier。

最明顯的例子：

| split | id | target | prediction | abs_error | selected_count |
|---|---:|---:|---:|---:|---:|
| val | 109586 | 12651 | 3924952.80 | 3912301.80 | 1 |
| val | 118743 | 14425 | 931980.80 | 917555.80 | 1 |
| val | 108450 | 15694 | 617414.60 | 601720.60 | 1 |
| test | 133844 | 161635 | 1228901.60 | 1067266.60 | 1 |
| test | 148080 | 16851 | 879430.40 | 862579.40 | 0 |
| test | 153482 | 271 | 474554.66 | 474283.66 | 1 |

prediction max 與 target max：

| split | prediction max | target max | prediction p99 | target p99 |
|---|---:|---:|---:|---:|
| val | 3924952.80 | 231528.90 | 142194.59 | 231528.90 |
| test | 1228901.60 | 231528.90 | 141364.07 | 202613.90 |

若只把 log prediction clamp 到 train target log 範圍再反轉：

| split | raw_R2 | clipped raw_R2 |
|---|---:|---:|
| val | -2.7417 | 0.2462 |
| test | -0.4927 | 0.0601 |

這說明負 R2 的直接原因就是少數極端高估值；但 clamp 後 R2 仍不高，代表模型本體還沒有學到足夠穩定的 signal。

## meanScore 為何變負

`meanScore` 沒有 expm1 爆炸問題，主要是 underfit 與 split shift。

| split | target_mean | prediction_mean | train_mean | R2 |
|---|---:|---:|---:|---:|
| val | 61.6049 | 59.1978 | 59.0320 | 0.1626 |
| test | 65.4302 | 59.4302 | 59.0320 | -0.2385 |

test split 的 `meanScore` 平均比 train 高約 6.4 分，但模型 prediction mean 仍約 59.4，幾乎等於 train mean。這代表 final model 對高分 test distribution 沒有跟上。

高誤差樣本多是低分被預測成 60 多，或高分被預測太低：

| split | id | target | prediction | abs_error | selected_count |
|---|---:|---:|---:|---:|---:|
| test | 126370 | 27 | 65.761 | 38.761 | 1 |
| test | 172016 | 28 | 66.454 | 38.454 | 8 |
| test | 185645 | 27 | 65.069 | 38.069 | 4 |
| test | 138065 | 74 | 39.825 | 34.175 | 9 |

## RRCP_silver 訊號偏弱

run 35 的 RRCP_silver 分布不是全 0，但非常接近 0，且約半數 item 被選入：

| target | split | mean | median | positive ratio | selected_count mean | empty rows |
|---|---|---:|---:|---:|---:|---:|
| popularity | train | 0.01533 | 0.01010 | 0.52734 | 5.273 | 375 |
| popularity | val | 0.01594 | 0.01099 | 0.52632 | 5.263 | 127 |
| popularity | test | 0.01254 | 0.00824 | 0.51931 | 5.193 | 134 |
| meanScore | train | -0.00236 | -0.00213 | 0.49060 | 4.906 | 574 |
| meanScore | val | 0.00550 | 0.00370 | 0.50703 | 5.070 | 175 |
| meanScore | test | 0.00410 | 0.00400 | 0.50855 | 5.086 | 168 |

目前 threshold 是 `0.0`，所以只要 RRCP 略大於 0 就被選入。因為分布集中在 0 附近，這個 selection 很容易變成噪聲閾值，而不是穩定的「有幫助 retrieval」判斷。

用 retrieved labels 做簡單 heuristic 時也看到 RRCP weighting 沒有帶來改善：

| target | heuristic | val_R2 | test_R2 |
|---|---|---:|---:|
| popularity | retrieved mean | 0.0073 | -0.0283 |
| popularity | RRCP selected mean | 0.0100 | -0.0669 |
| popularity | RRCP weighted | -0.0101 | -0.0881 |
| meanScore | retrieved mean | 0.0090 | -0.6515 |
| meanScore | RRCP selected mean | -0.0717 | -0.8435 |
| meanScore | RRCP weighted | -0.0755 | -0.8726 |

這表示目前 RRCP_silver 不是穩定增益來源，反而可能讓 final prediction head 學到不穩定權重。

## 與 SKAPP 原始碼的關鍵差距

已對照：

```text
baseline_refer/skapp-main/src/RRCP/predict_model.py
baseline_refer/skapp-main/src/RRCP/RRCP.py
baseline_refer/skapp-main/src/RRCP_prediction_variable_lenth.py
baseline_refer/skapp-main/src/graph_attention.py
baseline_refer/skapp-main/src/graph_variable_length.py
```

run 35 已有的構造：

| SKAPP 構造 | run 35 狀態 |
|---|---|
| all-items model | 已有 |
| dissembled/single-item model | 已有 |
| RRCP_silver | 已有 |
| threshold filtering | 已有 |
| variable-length selected retrieval | 已用 mask 近似 |
| GraphLearner-style graph propagation | 已有 cosine graph + graph convolution 近似 |
| CXMI/RRCP weighted attention head | 已有權重聚合近似 |

仍與原始碼不同、且可能影響數值穩定的地方：

| 差距 | 影響 |
|---|---|
| 原始碼 `feature_dim=768`，run 35 預設 `d_model=128` | 模型容量與原始架構不同；graph/attention representation 可能不足或失真 |
| 原始碼 `num_heads=8`，run 35 all-items/single 使用 4 heads，final 用加權聚合而非完整同構 attention | 對齊度仍不足 |
| 原始碼的 retrieved set 很大，RRCP source 中 `target_num=500`；run 35 目前 top_k=10 | RRCP 選樣空間太小，RRCP 分數較難穩定 |
| 原始任務是 social-media UGC popularity，本專案是 anime pre-release popularity | 缺少 user/social context，retrieved label 的意義不同 |
| run 35 沒有加入 project metadata/RAG aggregate features | 失去本專案最強的 tabular/retrieval floor，final model 只靠 query text/image + retrieved text/image/label |
| 沒有 output calibration 或 log prediction clipping | popularity 在 expm1 後容易被少數高估值拖垮 R2 |

## 現階段判斷

`C3-ProjectInputSKAPPFull` 目前不能解讀成「SKAPP 方法在本專案無效」。比較精確的說法是：

```text
SKAPPFull 的第一版完整構造已跑通，
但 RRCP_silver 訊號弱、final head 未校正，
且 project-input adaptation 缺少 metadata floor，
所以 performance 還不能作為最終 C3 reference baseline。
```

目前 C3 的 performance row 仍應該是：

```text
C3-RAG-Selective-XGB
```

而 `C3-ProjectInputSKAPPFull` 應定位為：

```text
structure-complete diagnostic row，等待穩定化與原始碼對齊補強。
```

## 下一步修正順序

### Step 1：先補診斷輸出

在 `run_c3_skapp_full.py` 增加：

1. all-items model 的 train/val/test metrics。
2. single/dissembled model 的 train/val/test metrics。
3. RRCP_silver summary JSON。
4. selected_count、empty rows、RRCP quantiles。
5. popularity log-space metrics 與 clipped raw-space metrics。

目的：先知道 RRCP_silver 是因為 all-items model 弱，還是 single-item model 弱。

### Step 2：先做不改架構的穩定化

1. popularity 的 raw prediction 評分前新增 log prediction clamp，範圍至少不得超過 train log target min/max。
2. 加入 validation calibration：用 val prediction 學一個簡單 linear calibration，再套到 test。
3. threshold 不固定 0，改測 top-m / per-row quantile threshold。

這些不會讓它變成 final SOTA，但可以確認負 R2 是否只是輸出校正問題。

## 2026-05-20 Post-hoc 穩定化試算

已用 run 35 既有 prediction 做不重訓試算，輸出存在本機 ignored result：

```text
.exp/baseline/results/35/c3_skapp_full_posthoc_stabilization_run35.json
```

試算結果：

| target | method | val_R2 | test_R2 | test_MAE | 解讀 |
|---|---|---:|---:|---:|---|
| popularity | raw | -2.7417 | -0.4927 | 14668.1228 | 原始 run 35 |
| popularity | clip to train log range | 0.2462 | 0.0601 | 13920.3988 | 負號可被 outlier clamp 修掉 |
| popularity | val log-linear calibration + clip | 0.1699 | 0.1295 | 12765.8056 | test 有改善，但仍不是強結果 |
| meanScore | raw | 0.1626 | -0.2385 | 9.8063 | 原始 run 35 |
| meanScore | val linear calibration | 0.2227 | 0.0098 | 8.6655 | 負號可被校正拉回，但仍弱 |

這個試算確認：

```text
1. popularity 的負 R2 確實主要來自 expm1 後的極端 outlier。
2. meanScore 的負 R2 確實主要來自 prediction mean 偏低與 split shift。
3. clamp/calibration 只能把負號修回正值，不能把 C3Full 變成可報告主結果。
4. 下一步必須診斷 all-items model、single/dissembled model 與 RRCP_silver 本身。
```

同時已在 `src/reference_baseline_branch/run_c3_skapp_full.py` 補上未來正式 run 的診斷輸出：

```text
c3_skapp_full_diagnostics_{target}.json
```

內容包含：

```text
all-items model metrics
single-item model metrics
RRCP_silver quantiles / selected_count / empty rows
final model raw-space metrics
final model model-space metrics
popularity clipped raw-space metrics
prediction/target distribution summaries
```

## 2026-05-21 Diagnostics Run 結論

已重跑兩個 diagnostics run：

```text
.exp/baseline/results/40  # threshold_of_rrcp = 0.0
.exp/baseline/results/41  # threshold_of_rrcp = 0.1
```

run 40 結果：

| target | val_MAE | val_R2 | val_Spearman | test_MAE | test_R2 | test_Spearman | test_log_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 14296.1232 | 0.2359 | 0.7868 | 13972.6180 | -0.0168 | 0.7440 | 1.2213 |
| meanScore | 8.2889 | 0.2146 | 0.4805 | 9.2430 | -0.1137 | 0.3730 |  |

run 41 結果：

| target | val_MAE | val_R2 | val_Spearman | test_MAE | test_R2 | test_Spearman | test_log_MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 14315.0579 | 0.2007 | 0.7587 | 13637.8418 | 0.0260 | 0.7194 | 1.3127 |
| meanScore | 8.5207 | 0.1668 | 0.4780 | 9.6808 | -0.2179 | 0.3713 |  |

### 最重要發現：final RRCP head 比 all-items model 差

run 40 的分段 diagnostics：

| target | stage | val_R2 | test_R2 | test_MAE |
|---|---|---:|---:|---:|
| popularity | all-items raw-space | 0.3042 | 0.1796 | 11993.5534 |
| popularity | final RRCP raw-space | 0.2359 | -0.0168 | 13972.6180 |
| meanScore | all-items | 0.3050 | 0.0258 | 8.5072 |
| meanScore | final RRCP | 0.2146 | -0.1137 | 9.2430 |

這代表 C3Full 差的主要斷點不是「一開始完全學不到」，而是：

```text
all-items model 已有一定訊號；
經過 single-item/RRCP_silver/threshold/final head 後，訊號反而下降。
```

### single-item model 明顯泛化差

run 40 training log 顯示：

```text
popularity single-item:
  train_loss 0.37909 -> 0.13137
  val_loss   0.50152 -> 0.53873

meanScore single-item:
  train_loss 0.57647 -> 0.21331
  val_loss   0.85727 -> 0.91146
```

single-item 是 RRCP_silver 的核心來源。它 train loss 下降很多，但 val loss 沒改善，
代表它產生的 with/without retrieval contribution 估計不穩，後續 RRCP_silver 很容易變成 noisy signal。

### threshold = 0.1 不是根治

threshold 從 `0.0` 提到 `0.1` 後，selected count 從約 5 個降到約 3 個：

| target | run | threshold | test selected mean | test empty rows |
|---|---:|---:|---:|---:|
| popularity | 40 | 0.0 | 5.135 | 134 |
| popularity | 41 | 0.1 | 3.039 | 691 |
| meanScore | 40 | 0.0 | 5.002 | 167 |
| meanScore | 41 | 0.1 | 3.062 | 734 |

但 performance 沒有真正改善：

```text
popularity:
  test_R2 -0.0168 -> 0.0260
  test_Spearman 0.7440 -> 0.7194
  test_log_MAE 1.2213 -> 1.3127

meanScore:
  test_R2 -0.1137 -> -0.2179
```

所以問題不是單純 threshold 太低，而是 RRCP_silver 的排序/強度本身不可靠。

### 目前最精確的原因描述

```text
C3-ProjectInputSKAPPFull 效果差，是因為 all-items model 仍有可用訊號，
但 single-item model 泛化差，導致 RRCP_silver contribution estimation 不可靠。
final RRCP head 再依照這個 noisy RRCP 做 variable-length selection 與 CXMI weighting，
反而把 all-items 已學到的訊號破壞掉。

此外 popularity 還有 expm1 raw-scale outlier 問題，
meanScore 則有 train/test target mean shift 問題。
```

### 下一個真正該修的點

不要再只調 threshold。下一步應優先修 RRCP 來源：

1. 讓 single-item/dissembled model 正規化更強：
   - dropout
   - smaller d_model
   - stronger weight decay
   - earlier stopping by val loss
   - 或先凍結 query/retrieved projections，降低 overfit
2. 產生 RRCP_silver 時改用 validation-calibrated single/all predictions。
3. 改成 top-m selection 或 per-row quantile selection，而不是固定 threshold。
4. final head 加 residual/fallback：讓 final prediction 同時保留 all-items prediction，不要完全依賴 RRCP-selected context。
5. 若要對本專案 EXP2 有用，另做一條 `metadata + SKAPPFull` project-aligned branch，補回 metadata floor。

## 2026-05-21 Regularized Run 結論

為了避免 baseline 被質疑「為了分數改到不像原論文」，這次只做訓練穩定化，
不改 RRCP 公式、不改 retrieved tensor schema、不改 Graph/RRCP attention 主結構。

程式新增參數：

```text
--dropout
--single-dropout
--variant-label
```

執行設定：

```text
.exp/baseline/results/42
baseline_id = C3-ProjectInputSKAPPFull-Regularized
dropout = 0.2
single_dropout = 0.4
weight_decay = 0.0005
patience = 4
threshold_of_rrcp = 0.0
```

結果比較：

| run | baseline_id | target | val_R2 | val_Spearman | test_MAE | test_R2 | test_Spearman | test_log_MAE |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 40 | `C3-ProjectInputSKAPPFull` | popularity | 0.2359 | 0.7868 | 13972.6180 | -0.0168 | 0.7440 | 1.2213 |
| 41 | `C3-ProjectInputSKAPPFull` threshold 0.1 | popularity | 0.2007 | 0.7587 | 13637.8418 | 0.0260 | 0.7194 | 1.3127 |
| 42 | `C3-ProjectInputSKAPPFull-Regularized` | popularity | 0.1601 | 0.7717 | 12941.9438 | 0.0855 | 0.7335 | 1.2702 |
| 40 | `C3-ProjectInputSKAPPFull` | meanScore | 0.2146 | 0.4805 | 9.2430 | -0.1137 | 0.3730 |  |
| 41 | `C3-ProjectInputSKAPPFull` threshold 0.1 | meanScore | 0.1668 | 0.4780 | 9.6808 | -0.2179 | 0.3713 |  |
| 42 | `C3-ProjectInputSKAPPFull-Regularized` | meanScore | 0.2196 | 0.4913 | 9.3310 | -0.1212 | 0.3835 |  |

分段 diagnostics：

| target | run | all-items test_R2 | single-item val_R2 | final test_R2 | RRCP test selected mean |
|---|---:|---:|---:|---:|---:|
| popularity | 40 | 0.1796 | 0.5840 | -0.0168 | 5.135 |
| popularity | 42 | 0.1834 | 0.6077 | 0.0855 | 5.068 |
| meanScore | 40 | 0.0258 | 0.1247 | -0.1137 | 5.002 |
| meanScore | 42 | -0.0129 | 0.1808 | -0.1212 | 5.083 |

解讀：

```text
Regularization 對 popularity 有小幅幫助：
single-item val_R2 變好，final test_R2 從 -0.0168 升到 0.0855，
test_MAE 從 13972.6180 降到 12941.9438。

但 meanScore 幾乎沒有改善，甚至 all-items test_R2 下降。
因此 regularization 只能說明「訓練穩定化有一點效果」，
不能讓 SKAPPFull 成為強 baseline。
```

目前應採用的定位：

```text
C3-ProjectInputSKAPPFull-Regularized 可以保留為 stability diagnostic variant。
正式 reference performance row 仍應使用 C3-RAG-Selective-XGB。
C3Full / C3Full-Regularized 只能說明我們已完成 SKAPP structure-complete reconstruction，
並且檢查過 RRCP 在本專案 anime pre-release setting 下不穩定。
```

建議停損：

```text
若不引入 all-items residual、metadata branch、或大幅重設 retrieval_num/top_k，
單靠 regularization / threshold 很難讓 C3Full 追上 C3-RAG-Selective-XGB。
而 residual / metadata branch 會變成本專案改良版，不應再稱純 SKAPP reproduction。
```

### Step 3：再對齊 SKAPP 原始架構

1. 將 `d_model` 提升到 768 或至少跑 256/512/768 對照。
2. all-items/single/final 的 attention heads 對齊原始碼 `num_heads=8`。
3. final model 的 graph_attention 與 `RRCP_prediction_variable_lenth.py` 更接近原始碼。
4. top_k 從 10 擴到 20/50，若算力允許再接近原始碼更大的 retrieval pool。

### Step 4：補回 project metadata floor

若目標是讓 C3 對本專案 EXP2 有用，不能只保留 SKAPP 的 query/retrieved text-image-label tensor。需要新增一條 project-aligned full model：

```text
metadata + query text + query image + retrieved text/image/label + RRCP/CXMI
```

否則它會缺少目前 `C3-RAG-Selective-XGB` 能用到的 metadata 與 aggregate RAG signal，對 EXP2 主框架的比較價值不足。
