# 外部評估總結

本文件整理 `src_2` 多模態融合模型第一版可重現的 MAL-only 外部資料集評估。

## 目的

外部評估的目的，是檢查只用內部 AniList 資料訓練出的模型，面對不在內部
train/validation/test split 內的 MAL 動畫時，是否仍保留有效的排序訊號。

正式對齊與排除只使用穩定 ID。動畫名稱或模糊 title matching 不作為正式實驗依據。

## 資料集選擇

主要外部考卷：

- 來源：`outtestdataset/MyAnimeList 2025/mal_anime.csv`
- 採用原因：同時包含 MAL ID、cover image URL、description、metadata、`members`
  與 `score`
- 排除方式：凡是能透過 MAL/AniList crosswalk 對回內部 AniList universe 的 rows，
  都從 MAL-only 外部考卷中排除

先前的 `MyAnimeList Anime & Manga Dataset (July 2025)` 仍保留作為 label
sanity check，但因為沒有 image URL，不作為 full multimodal 外部主考卷。

## 最終外部考卷

下載 cover 圖片並移除本機圖片缺失 rows 後，得到兩份 local-ready external split：

| Split | Rows | Targets | 說明 |
|---|---:|---|---|
| `mal2025_popularity_local_ready` | 3,765 | MAL `members` | 主要 popularity-only 外部考卷 |
| `mal2025_dual_local_ready` | 1,202 | MAL `members`, MAL `score * 10` | 較小但可同時評估 popularity 與 score 的外部考卷 |

原本 image-ready 候選數量是 3,798 筆 popularity rows 與 1,209 筆 dual-target
rows。差額來自少數 MAL cover URL 回傳 404 或無法成功下載成可用本機圖片。

## 特徵轉換策略

adapter 會把外部資料轉成目前 `src_2` 模型可讀的輸入契約：

- metadata CSV 會輸出到 `src_2/data/dataset/`
- 外部 numeric ID 使用 `900000000 + mal_id`，避免和 AniList ID 撞號
- text embeddings 由 MAL description 產生
- image embeddings 使用 MAL cover image 產生
- 第一版外部評估中，banner 與 YOLO branch 視為 missing modality
- RAG features 由內部 AniList Qdrant collection 檢索產生

## Run02 結果

第一版完成的外部推論使用 run02 checkpoints。

| Exam | Metric | Value |
|---|---|---:|
| popularity-only，3,765 rows | Spearman(prediction, MAL `members`) | 0.4709 |
| popularity-only，3,765 rows | Pearson(log prediction, log MAL `members`) | 0.5482 |
| dual-target，1,202 rows | Spearman(popularity prediction, MAL `members`) | 0.5495 |
| dual-target，1,202 rows | MAE(meanScore prediction, MAL `score * 10`) | 7.5086 |
| dual-target，1,202 rows | Spearman(meanScore prediction, MAL `score * 10`) | 0.6079 |

這裡刻意不回報 raw popularity MAE，因為 AniList popularity 與 MAL members
屬於不同平台的累積 count scale。外部比較較適合使用 ranking 指標與 log-scale
相關性。

## 解讀

這不是單純把內部資料換一個 label 重算，而是第一版有效的外部泛化檢查：

- 考卷 rows 已經用 ID-based exclusion 排除內部 AniList 已知資料
- 模型使用的是重新產生的外部 text、image、metadata 與 RAG features
- 結果顯示模型在外部 MAL-only rows 上仍有中等正相關的排序轉移能力，尤其
  dual-target split 的 score Spearman 達 0.6079

因此，這份結果可以作為第一版外部驗證報告。若後續模型架構或最佳 checkpoint
更新，應使用同一套外部考卷與腳本重跑 final paper 數字。

## 重現指令

將外部資料集放在 `outtestdataset/` 後，從 repository root 執行：

```bash
python scripts/external/prepare_external_evaluation_assets.py
python scripts/external/download_external_images.py \
  --exam-csv data/external_transformed/mal2025_image_mal_only_popularity_exam.csv \
  --sleep 0
python scripts/external/prepare_external_local_ready_exams.py
python scripts/external/prepare_external_model_inputs.py
python scripts/external/build_external_embeddings.py \
  --splits mal2025_popularity_local_ready mal2025_dual_local_ready \
  --modality text
python scripts/external/build_external_embeddings.py \
  --splits mal2025_popularity_local_ready mal2025_dual_local_ready \
  --modality image \
  --image-model-path results/01/best
python src_2/RAG/rag_builder.py
python src_2/RAG/rag_query.py \
  --splits mal2025_popularity_local_ready mal2025_dual_local_ready
python scripts/external/run_external_inference.py \
  --run-id 02 \
  --split mal2025_dual_local_ready \
  --output-prefix run02_mal2025_dual_local_ready
python scripts/external/run_external_inference.py \
  --run-id 02 \
  --split mal2025_popularity_local_ready \
  --targets popularity \
  --output-prefix run02_mal2025_popularity_local_ready
```
