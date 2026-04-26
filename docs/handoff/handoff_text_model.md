# 交接文件：文字處理組

本文件提供文字處理組在現有資料流程上的直接接手指南。

## 1) 任務定位

- 任務：建立文字分支的可重現特徵流程，支援 `popularity` 與 `meanScore` 雙迴歸。
- 範圍：以 pre-release 可取得資訊為主，不納入播後留言與社群互動。
- 關聯研究問題：RQ2（文字語義加入後是否帶來穩定增益）。

## 2) 主要輸入資料

- 來源檔：`data/processed/anilist_anime_multimodal_input_v1.csv`
- 主要文字欄位：
  - `description`（主文字訊號）
  - `title_romaji`, `title_english`（輔助文字訊號）
- 分割欄位：`split_pre_release_effective`
- 目標欄位：`popularity`, `meanScore`

## 3) 前處理建議（第一版）

- 清理：
  - 統一空值與空字串
  - 移除過長重複標記（URL、模板文字）
- Tokenization：
  - Baseline 先使用 Sentence-Transformer / BERT family tokenizer
- 向量化：
  - 第一版採 frozen embedding + 線性/MLP 回歸頭
  - 第二版再嘗試 end-to-end fine-tuning

## 4) 交付與輸出規格

- 產出建議：
  - `artifacts/text_embeddings_{split}.parquet`
  - `reports/text_branch_metrics.json`
- 指標：
  - `MAE`, `RMSE`, `Spearman`（兩個 target 都要回報）
- 最少要有：
  - train/val/test 同一套流程可重跑
  - 固定 random seed

## 5) 代辦清單（接手後順序）

1. 建立文字清理與 tokenization 腳本。
2. 產出 split 對齊的 embedding 快取。
3. 建立文字-only baseline（兩個回歸目標）。
4. 回填文字分支實驗結果到論文圖表/表格。

## 6) 風險與注意事項

- `description` 可用率雖高但非 100%，需與 `has_text_description` 一起做缺模態策略。
- 不可打破既有 split，避免資料洩漏。
- 若使用外部預訓練模型，需在報告內記錄版本與下載時間。
