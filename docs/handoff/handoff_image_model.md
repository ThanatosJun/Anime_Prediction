# 交接文件：圖片處理組

本文件提供圖片處理組在現有資料流程上的直接接手指南。

## 1) 任務定位

- 任務：建立影像分支特徵流程，支援 `popularity` 與 `meanScore` 雙迴歸。
- 範圍：以 pre-release 可取得視覺素材為主。
- 關聯研究問題：RQ2（語義影像特徵相對於純 metadata 是否有增益）。

## 2) 主要輸入資料

- 來源檔：`data/processed/anilist_anime_multimodal_input_v1.csv`
- 圖像欄位：
  - `coverImage_medium`（主圖）
  - `bannerImage`（輔助圖）
  - `trailer_thumbnail`（可選）
- 分割欄位：`split_pre_release_effective`
- 目標欄位：`popularity`, `meanScore`

## 3) 前處理建議（第一版）

- 圖像抓取：
  - 下載素材至本機快取（避免重複抓取）
  - 記錄失敗 URL 與重試次數
- 影像轉換：
  - resize、normalize、center crop
  - 保持 train/val/test 同分佈策略
- 特徵抽取：
  - Baseline 使用 frozen backbone（ResNet50 / CLIP ViT）
  - 先做 feature extraction，再接簡單回歸頭

## 4) 交付與輸出規格

- 產出建議：
  - `artifacts/image_embeddings_{split}.parquet`
  - `reports/image_branch_metrics.json`
- 指標：
  - `MAE`, `RMSE`, `Spearman`
- 最少要有：
  - URL 下載錯誤報表
  - 影像可用率與 drop 策略紀錄

## 5) 代辦清單（接手後順序）

1. 建立圖片下載與快取腳本。
2. 建立影像 embedding 產生流程（split 對齊）。
3. 建立 image-only baseline（兩個回歸目標）。
4. 回填圖片分支結果到論文比較表。

## 6) 風險與注意事項

- `bannerImage` 缺失率高於 `coverImage_medium`，建議主流程以 `coverImage_medium` 為核心。
- 不可跨 split 共用 augmentation 統計，避免洩漏。
- 若圖片來源更新，需重建對應 embedding 並記錄快照版本。
