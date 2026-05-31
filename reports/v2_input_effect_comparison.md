# V2 Input Effect Comparison

## 結論

V2 目前不是單純「新版 image embedding」。在 baseline 端，V2 的實際變化是：

1. metadata/split 從舊 baseline 的 `data/fussion/post2000` 改為 `data/fussion/fusion_meta_clean_{split}_v2.csv`
2. train set 從 9,583 筆增加到 13,321 筆，新增的 3,738 筆主要是 1940-1999 年作品
3. image embedding 仍使用 `src/fussion_branch/embedding/image/image_embeddings_{split}.parquet`，不是 `src_2` 重新產出的 embedding
4. V2 CSV 本身相對 full `fusion_meta_clean_{split}.csv` 只移除無真實封面圖資料：train 55 筆、holdout_unknown 2 筆，val/test 沒變

因此目前 V2 baseline 結果不能解讀成「新版 image embedding 帶來的效果」，只能解讀成「full V2 metadata/split 對齊後的 baseline 效果」。

## 新資料到底更新了什麼

相對原始 full CSV：

| split | full | v2 | removed | 說明 |
|---|---:|---:|---:|---|
| train | 13,376 | 13,321 | 55 | 移除無真實封面圖/default 圖資料，年份 1949-1974 |
| val | 2,918 | 2,918 | 0 | 無變化 |
| test | 3,087 | 3,087 | 0 | 無變化 |
| holdout_unknown | 943 | 941 | 2 | 移除無真實封面圖資料，年份 1968、1985 |

相對舊 baseline 使用的 `post2000`：

| split | post2000 | v2 | v2 extra |
|---|---:|---:|---:|
| train | 9,583 | 13,321 | +3,738 |
| val | 2,918 | 2,918 | 0 |
| test | 3,087 | 3,087 | 0 |

也就是目前 baseline 前後差異主要是「train 訓練分布變了」，不是 val/test 變了。

## Baseline 與之前輸入相比

完整逐列比較在 `reports/reference_baseline_v2_vs_previous.csv`。

主要觀察：

| baseline | target | old test MAE | v2 test MAE | 變化 |
|---|---|---:|---:|---:|
| F1-RF-Meta | popularity | 8590.0532 | 8551.7168 | -38.3364 |
| F1-RF-Meta | meanScore | 7.9541 | 8.0179 | +0.0638 |
| F2-XGB-Concat | popularity | 9588.2590 | 9688.3006 | +100.0416 |
| F2-XGB-Concat | meanScore | 8.3391 | 8.5473 | +0.2082 |
| I1-XGB-ImageEmb | popularity | 13815.0865 | 13928.1819 | +113.0954 |
| I1-XGB-ImageEmb | meanScore | 9.4042 | 9.4161 | +0.0119 |
| C1-Armenta-ProjectInputProxy | popularity | 11951.3799 | 11672.2261 | -279.1538 |
| C1-Armenta-ProjectInputProxy | meanScore | 8.7567 | 9.2523 | +0.4956 |
| C2-ProjectInputCrossAttention | popularity | 12755.1921 | 11044.7140 | -1710.4781 |
| C2-ProjectInputCrossAttention | meanScore | 8.0600 | 8.4384 | +0.3784 |
| C1-Armenta-ProjectInputReconstruction | popularity | 10719.7513 | 10501.5398 | -218.2115 |
| C1-Armenta-ProjectInputReconstruction | meanScore | 9.0250 | 10.5367 | +1.5117 |
| C2-ProjectInputCTNNReconstruction | popularity | 10151.2161 | 10448.2886 | +297.0725 |
| C2-ProjectInputCTNNReconstruction | meanScore | 8.1751 | 8.3066 | +0.1315 |
| C3-RAG-Selective-XGB | popularity | 9782.2338 | 9520.2222 | -262.0116 |
| C3-RAG-Selective-XGB | meanScore | 8.0914 | 8.3090 | +0.2176 |
| C3-ProjectInputSKAPPProxy-XGB | popularity | 10239.2909 | 10121.7524 | -117.5385 |
| C3-ProjectInputSKAPPProxy-XGB | meanScore | 8.1715 | 8.2630 | +0.0915 |
| C3-ProjectInputSKAPPGraphProxy | popularity | 11501.8681 | 11512.0077 | +10.1396 |
| C3-ProjectInputSKAPPGraphProxy | meanScore | 8.1448 | 8.5741 | +0.4293 |

整體趨勢：

- popularity：部分方法變好，尤其 C2 cross/recurrent 類、C3 selective 類；但不是全面改善
- meanScore：多數方法變差，尤其 C1/C2 深度 proxy
- image-only：幾乎沒有改善，且 Spearman 略降，因為目前 image parquet 不是新版 `src_2` 產物

## 自身 Fusion 框架相比

目前 repo 內沒有 `.exp/fussion/results` raw result；可查到的是 `src/fussion_branch/README.md` 的實驗表。

README 中 V2 對應的是 Run15：

| target | reference run | old/test | V2 Run15/test | 變化 |
|---|---|---:|---:|---:|
| popularity log_MAE | Run11 | 0.9766 | 1.0112 | +0.0346 |
| meanScore MAE | Run11 | 8.0691 | 8.0865 | +0.0174 |

依照 README 表格本身，Run15 並沒有優於 Run11：

- popularity test log_MAE 從 0.9766 變 1.0112，變差
- meanScore test MAE 從 8.0691 變 8.0865，微幅變差
- val 指標也不是 Run15 最佳；表中 Run10/Run11 的 popularity val log_MAE 更低，Run11 的 meanScore val MAE 也更低

所以 README 目前「Run15 在兩個 target 的 val 均達最佳」這句與表格數字不一致，應標記為待修正。

## 後續判定

若正式論文要比較 V2：

1. 先決定 V2 image 是否必須使用 `src_2/model/best` 重新產生 embedding
2. 若是，就要產出新的 `image_embeddings_{split}.parquet` 到獨立 V2 目錄
3. baseline 與 Fusion 主框架都指向同一份 V2 image embedding
4. 再重新跑 baseline 和 Fusion，才可以討論「新版 image/V2 pipeline」的效果
