# 外部評估 Adapter 交接

本文件定義模型架構穩定後，MAL-only 外部考卷如何接到推論流程。

## 1. 目標

建立 MAL-only inference adapter，讓模型可以對不在內部 AniList 訓練資料中的 MAL anime 做預測，並與外部答案比對。

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

建議以 full multimodal adapter 為主，分三段落地。

### A. Image asset materialization

用途：把 exam CSV 內的 cover URL 變成本地圖片資產。

需求：

1. 讀取 `external_cover_image_url`。
2. 下載到 `external_cover_image_path`。
3. 失敗列保留，但在 image embedding 中標為 `has_cover=0`。
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
4. banner/yolo branch 以 zero vector + missing mask 處理，或由 cover 產生 YOLO crops。
5. RAG 以 `release_year` + `release_quarter` 做時間過濾。
6. 對有 `resolved_anilist_id` 的 rows 做 self-exclusion；沒有則不做 self-exclusion。

可使用：

```bash
python scripts/external/build_external_embeddings.py \
  --splits mal2025_popularity_local_ready mal2025_dual_local_ready \
  --modality both
python src_2/RAG/rag_query.py \
  --splits mal2025_popularity_local_ready mal2025_dual_local_ready
```

目前本機已完成兩個 external split 的 text embeddings。image embeddings 仍需
`src_2/component_image/model-image/best` 的 Swin 權重；沒有該目錄時，
`build_external_embeddings.py --modality image` 會明確失敗並提示缺少模型。

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
  --split mal2025_dual_local_ready \
  --output-prefix run02_mal2025_dual_local_ready
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

## 6. 目前不做的事

1. 不用 title matching 建立正式對齊。
2. 不把 Largest MAL User Dataset 併入主流程。
3. 不在模型架構穩定前綁死 full multimodal inference code。
4. 不覆蓋既有 `data/processed` train/val/test 檔案。
