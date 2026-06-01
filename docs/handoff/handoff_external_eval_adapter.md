# 外部評估 Adapter 交接

本文件定義 MAL-only 外部考卷如何接到目前 `src_2` 推論流程，並記錄
第一版 run02 外部評估狀態。

## 1. 目標

建立 MAL-only inference adapter，讓模型可以對不在內部 AniList 訓練資料中的 MAL anime 做預測，並與外部答案比對。

目前已完成第一版 adapter 與 run02 外部推論。後續若模型 checkpoint 或
架構更新，應重跑同一套 external split，而不是重新定義考卷。

主考卷：

- `data/external_transformed/mal2025_image_mal_only_dual_target_exam.csv`

補充考卷：

- `data/external_transformed/mal2025_image_mal_only_popularity_exam.csv`

`mal_july2025_*` 保留為外部 label sanity check。該來源沒有圖片欄位，不作為 full multimodal 主考卷。

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
18. `external_cover_image_url`
19. `external_cover_image_path`

可選欄位：

1. `aodb_anilist_id`
2. `external_popularity_rank`
3. `external_score_0_10`
4. `external_scored_by`
5. `coverImage_extraLarge`
6. `bannerImage`

## 3. 推論策略

已以 full multimodal adapter 為主，分三段落地。

### A. Image asset materialization

用途：把 exam CSV 內的 cover URL 變成本地圖片資產。

需求：

1. 讀取 `external_cover_image_url`。
2. 下載到 `external_cover_image_path`。
3. 失敗列輸出到 missing-local-image CSV，不納入正式 local-ready 評估。
4. banner 目前沒有來源，固定 missing。

可先使用：

```bash
python scripts/external/download_external_images.py \
  --exam-csv data/external_transformed/mal2025_image_mal_only_popularity_exam.csv \
  --sleep 0
python scripts/external/prepare_external_local_ready_exams.py
```

下載 popularity exam 會同時涵蓋 dual-target exam 的圖片。後續 adapter 應讀取
`*_local_ready.csv`，避免 404 圖片列混入正式 full multimodal 評估。

接著產生 `src_2` 可讀的 external split：

```bash
python scripts/external/prepare_external_model_inputs.py
```

這會輸出：

1. `src_2/data/dataset/fusion_meta_clean_mal2025_popularity_local_ready_v2.csv`
2. `src_2/data/dataset/fusion_meta_clean_mal2025_dual_local_ready_v2.csv`
3. `data/external_transformed/mal2025_popularity_local_ready_id_map.csv`
4. `data/external_transformed/mal2025_dual_local_ready_id_map.csv`

external split 使用 `900000000 + mal_id` 作為 numeric surrogate id，避免與 AniList ID 撞號。

### B. External feature generation

用途：產生目前模型需要的 metadata/text/image/RAG inputs。

需求：

1. 將 MAL metadata 欄位直接餵給目前 metadata encoder。
2. 用 `description` 產生 text embeddings。
3. 用下載的 cover 產生 cover image embeddings。
4. 第一版 banner/yolo branch 以 zero vector + missing mask 處理。
5. RAG 以 `release_year` + `release_quarter` 做時間過濾。
6. 對有 `resolved_anilist_id` 的 rows 做 self-exclusion；沒有則不做 self-exclusion。

可使用：

```bash
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
```

目前本機已完成兩個 external split 的 text embeddings、image embeddings
與 RAG returns。若審查者的 Swin 權重路徑不同，請用
`--image-model-path` 指定。

### C. Full multimodal inference

用途：用現有模型對 MAL-only rows 產生 prediction，並和外部答案比較。

需求：

1. 使用 `external_exam_id` 作為外部 row id。
2. 保留 `mal_id` 與 `resolved_anilist_id`。
3. 分別輸出 popularity 與 meanScore prediction。
4. 保留 inference profile，例如 `cover_only_missing_banner_yolo`。

可使用：

```bash
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

## 6. 第一版 run02 外部結果

1. `mal2025_popularity_local_ready`，3,765 rows：
   - popularity Spearman vs MAL `members`：0.4709。
   - popularity log MAE vs MAL `members`：1.0120。
   - popularity log R2 vs MAL `members`：0.2709。
   - popularity factor_acc_2x vs MAL `members`：0.4656。
   - popularity raw MAE diagnostic vs MAL `members`：3518.3324。
   - popularity log Pearson vs MAL `members`：0.5482。
2. `mal2025_dual_local_ready`，1,202 rows：
   - popularity Spearman vs MAL `members`：0.5495。
   - popularity log MAE vs MAL `members`：1.3910。
   - popularity log R2 vs MAL `members`：-0.4610。
   - popularity factor_acc_2x vs MAL `members`：0.3344。
   - popularity raw MAE diagnostic vs MAL `members`：7750.9359。
   - meanScore MAE vs MAL `score * 10`：7.5086。
   - meanScore R2 vs MAL `score * 10`：-1.0659。
   - meanScore acc_within_10pt vs MAL `score * 10`：0.7488。
   - meanScore Spearman vs MAL `score * 10`：0.6079。

完整報告見 `reports/external/external_evaluation_summary.md`。

## 7. 目前不做的事

1. 不用 title matching 建立正式對齊。
2. 不把 Largest MAL User Dataset 併入主流程。
3. 不覆蓋既有 `data/processed` train/val/test 檔案。
4. 不把 generated CSV、prediction、embedding、image assets commit 進 git。
