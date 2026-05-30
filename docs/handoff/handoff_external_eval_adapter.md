# 外部評估 Adapter 交接

本文件定義模型架構穩定後，MAL-only 外部考卷如何接到推論流程。

## 1. 目標

建立 MAL-only inference adapter，讓模型可以對不在內部 AniList 訓練資料中的 MAL anime 做預測，並與外部答案比對。

主考卷：

- `data/external_transformed/mal_july2025_mal_only_dual_target_exam.csv`

補充考卷：

- `data/external_transformed/mal_july2025_mal_only_popularity_exam.csv`

## 2. 輸入契約

MAL-only exam rows 使用 `external_exam_id` 作為推論 row id。

必要欄位：

1. `external_exam_id`
2. `mal_id`
3. `title_romaji`
4. `title_english`
5. `format`
6. `status`
7. `season`
8. `release_year`
9. `release_quarter`
10. `episodes_numeric`
11. `duration_minutes`
12. `source`
13. `genres`
14. `studios`
15. `description`
16. `external_popularity_members`
17. `external_score_0_100`

可選欄位：

1. `aodb_anilist_id`
2. `external_popularity_rank`
3. `external_score_0_10`
4. `external_scored_by`

## 3. 推論策略

建議分三階段接入。

### A. Metadata + text smoke test

用途：先驗證外部考卷是否能被模型接口吃進去。

需求：

1. 將 MAL metadata 轉成目前 metadata encoder 接受的欄位。
2. 用 `description` 產生 text embeddings。
3. image branch 使用 missing-image fallback 或關閉。
4. RAG branch 使用 fallback 特徵或關閉。

### B. Metadata + text + RAG

用途：測試沒有完整 image asset 時，RAG 是否仍能提升外部泛化。

需求：

1. 對有 `aodb_anilist_id` 的 rows 可以用 AniList ID 做 self-exclusion。
2. 對沒有 AniList ID 的 rows 不做 self-exclusion。
3. 以 `release_year` + `release_quarter` 做時間過濾。
4. 若 text embedding 不存在，回退到 global RAG fallback。

### C. Full multimodal

用途：模型架構穩定後的正式外部新資料推論。

需求：

1. 下載或解析 MAL cover image asset。
2. 產生 image embeddings。
3. 產生 text embeddings。
4. 對 image-missing rows 定義 consistent fallback。
5. 輸出雙目標 predictions。

## 4. 輸出契約

建議輸出 CSV：

- `data/external_transformed/mal_only_dual_target_predictions_<run>.csv`
- `data/external_transformed/mal_only_popularity_predictions_<run>.csv`

必要欄位：

1. `external_exam_id`
2. `mal_id`
3. `title_romaji`
4. `prediction_popularity`
5. `prediction_meanScore`
6. `external_popularity_members`
7. `external_score_0_100`
8. `external_popularity_rank`
9. `model_run_id`
10. `inference_profile`

## 5. 評估指標

Popularity：

1. Spearman against `external_popularity_members`
2. Spearman against negative `external_popularity_rank`
3. Pearson on `log1p(prediction)` vs `log1p(members)`
4. log-MAE against `members`

Score：

1. MAE against `external_score_0_100`
2. RMSE against `external_score_0_100`
3. Spearman against `external_score_0_100`

不要用 raw MAE 比較 AniList popularity prediction 與 MAL members，兩者是不同平台的 count scale。

## 6. 目前不做的事

1. 不用 title matching 建立正式對齊。
2. 不把 Largest MAL User Dataset 併入主流程。
3. 不在模型架構穩定前綁死 full multimodal inference code。
4. 不覆蓋既有 `data/processed` train/val/test 檔案。

