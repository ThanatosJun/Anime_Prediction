# 外部資料評估方法

本文件說明外部資料集如何接到目前 AniList 多模態預測框架，並定義後續外部考卷與評估規則。

## 1. 原則

外部資料只透過穩定 ID 對齊，不使用動畫名稱做主鍵。

- 首選：AniList ID
- 次選：MAL ID
- 禁用：title/name 模糊對齊作為正式實驗依據

原因是動畫名稱容易因劇場版、OVA、續作、重製版、語言別名而混淆。若未來需要 title matching，只能作為人工審核輔助，不納入自動實驗主流程。

## 2. 資料來源分工

### Anime Offline Database

用途：

1. 建立 AniList ID 與 MAL ID 的 crosswalk。
2. 補 `holdout_unknown` 的 `release_quarter`。
3. 作為 future-work metadata enrichment 來源。

不作為主要外部答案來源，因為它缺少可直接對應目前 `popularity` target 的欄位。

目前驗證：

1. AODB 原始列數：40,515。
2. unique AniList IDs：20,352。
3. unique MAL IDs：29,932。
4. 原本 `holdout_unknown`：943。
5. 可補回 release year + quarter：789。
6. 補後仍 unknown：154。

補回的 789 筆不覆蓋正式 processed 檔案，只輸出 future-work 檔案。

### MyAnimeList Anime & Manga Dataset July 2025

用途：

1. 外部 label sanity check。
2. 對齊後的 cross-platform label sanity check。
3. 建立 text/metadata-only MAL-only 外部新資料候選。

外部 label 對應：

1. `external_popularity_members`：MAL `members`，作為 external popularity count proxy。
2. `external_score_0_100`：MAL `score * 10`，粗略對齊 AniList `meanScore` 的 0-100 尺度。
3. `external_popularity_rank`：MAL popularity rank，數字越小越熱門，只用於 ranking diagnostics。

目前驗證：

1. MAL anime source rows：28,635。
2. 可對齊內部 AniList rows：19,090。
3. `external_eval_ready` rows：15,590。
4. MAL-only rows：9,545。
5. MAL-only dual-target rows：2,510。
6. MAL-only popularity-only rows：9,545。
7. 原始 CSV 沒有 image/cover/banner/thumbnail 欄位。

因此 July 2025 不作為 full multimodal 外部考卷，只保留作為外部答案與文字/metadata 檢查資料。

### MyAnimeList 2025

用途：

1. 正式 full multimodal MAL-only 外部考卷來源。
2. 提供 MAL ID、cover image URL、description、metadata、members、score。
3. 只透過 MAL ID + AODB/raw crosswalk 排除內部 AniList rows，不使用 title matching。

目前驗證：

1. source rows：19,931。
2. 有 cover image URL：19,544。
3. 有 cover URL、description、release year + quarter：19,283。
4. 可對齊內部 AniList 的 image-ready rows：15,485。
5. 保守 MAL-only image-ready popularity rows：3,798。
6. 保守 MAL-only image-ready dual-target rows：1,209。
7. 下載圖片後 local-ready popularity rows：3,765。
8. 下載圖片後 local-ready dual-target rows：1,202。

限制：

1. 只有 cover image URL，沒有 banner image。
2. YOLO crops 必須由下載後的 cover 圖重新產生。
3. 第一版外部評估將 banner 與 YOLO branch 視為 missing modality，由
   missing mask/zero vector 處理。

## 3. 外部考卷定義

### Cross-platform aligned evaluation

檔案：

- `data/external_transformed/mal_july2025_external_eval_contract.csv`

用途：

1. 驗證 AniList 與 MAL 的 label 是否一致。
2. 將既有 AniList test predictions 與 MAL label 比對。

這不是真正的新資料 inference，因為樣本已存在於內部 AniList 資料集中。

目前 sanity check：

1. AniList `popularity` vs MAL `members` Spearman：0.9757。
2. AniList `meanScore` vs MAL `score * 10` Spearman：0.9339。

解讀：MAL label 適合作為外部考卷答案，但不代表模型已經完成外部泛化驗證。

### MAL-only image-ready dual-target exam

檔案：

- `data/external_transformed/mal2025_image_mal_only_dual_target_exam.csv`

用途：

1. 主外部新資料考卷。
2. 同一批樣本同時評估 popularity 與 score。
3. 已支援 full multimodal feature generation 與 run02 外部推論。

目前狀態：

1. rows：1,209。
2. 有 `members`：1,209。
3. 有 `score * 10`：1,209。
4. 有 release year + quarter：1,209。
5. 有 text description：1,209。
6. 有 cover image URL：1,209。
7. local-ready rows：1,202。
8. 已產生 text/image/RAG features。
9. 已完成 run02 外部推論。

### MAL-only image-ready popularity-only exam

檔案：

- `data/external_transformed/mal2025_image_mal_only_popularity_exam.csv`

用途：

1. 補充實驗。
2. 只評估 popularity 泛化能力。
3. 樣本數比 dual-target exam 大，適合檢查人氣排序泛化。

目前狀態：

1. rows：3,798。
2. 有 `members`：3,798。
3. 有 release year + quarter：3,798。
4. 有 text description：3,798。
5. 有 cover image URL：3,798。
6. local-ready rows：3,765。
7. 已產生 text/image/RAG features。
8. 已完成 run02 外部推論。

## 4. Full multimodal adapter 狀態

MAL-only 外部資料不是內部 AniList rows，因此不能直接重用內部
train/val/test 的中間產物；但目前已補上 adapter，能將外部資料轉成
`src_2` 可讀的 external split。

目前已完成：

1. 下載 MAL cover image，並輸出 local-ready exam CSV。
2. 以 `900000000 + mal_id` 建立 surrogate numeric id，避免與 AniList ID 撞號。
3. 將 MAL metadata 轉成 `fusion_meta_clean_<split>_v2.csv`。
4. 從 MAL description 產生 text embeddings。
5. 從 MAL cover image 產生 image embeddings。
6. 對缺少的 banner/YOLO 分支以 missing mask/zero vector 處理。
7. 以內部 AniList Qdrant collection 產生 external RAG returns。
8. 用 `run_external_inference.py` 產生 prediction 與外部 metric JSON。

限制仍需在論文中說明：

1. 外部資料只有 cover，沒有 banner。
2. 第一版沒有重新生成 YOLO crop embedding。
3. RAG knowledge base 仍是內部 AniList universe，外部 rows 只作 query。

## 5. 第一版 run02 外部結果

run02 外部推論結果如下：

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

這是第一版可回報的外部泛化測試。若後續模型架構或最佳 checkpoint 變動，
應使用同一套 external split 與腳本重跑 final run。

## 6. 重跑方式

```bash
python scripts/external/prepare_external_evaluation_assets.py
```

若要重現 MAL 2025 full multimodal 外部考卷的圖片與 local-ready 篩選：

```bash
python scripts/external/download_external_images.py \
  --exam-csv data/external_transformed/mal2025_image_mal_only_popularity_exam.csv \
  --sleep 0
python scripts/external/prepare_external_local_ready_exams.py
python scripts/external/prepare_external_model_inputs.py
```

正式 full multimodal 評估應讀取 `*_local_ready.csv`，因為少數 MAL 圖片 URL 可能回傳 404。

產生 external split features 與 predictions：

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

`prepare_external_model_inputs.py` 會用 `900000000 + mal_id` 建立 numeric surrogate id，並輸出 id map sidecar，避免外部 MAL row 和內部 AniList row 撞號。

若要將既有 prediction 檔與 MAL aligned labels 比對：

```bash
python scripts/external/evaluate_external_predictions.py \
  --predictions-root ".exp/baseline/results/39/predictions/C2-ProjectInputCTNNDualVisualReconstruction" \
  --split test \
  --output-prefix run39_c2_dual_visual_mal_july2025_external
```

新版 `src_2/evaluate.py` 輸出也可直接評估：

```bash
python src_2/evaluate.py --target popularity --split test
python src_2/evaluate.py --target meanScore --split test

python scripts/external/evaluate_external_predictions.py \
  --predictions-root "src_2/runs/01" \
  --split test \
  --output-prefix run01_mal_july2025_external
```

評估腳本同時支援兩種 prediction 檔名與欄位格式：

1. 舊版：`<target>/<split>_predictions.csv`，欄位 `id,target,prediction`。
2. 新版：`<target>/pred_<split>.csv`，欄位 `id,pred,target`。
