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

1. 主要外部評估資料。
2. 對齊後的 cross-platform label sanity check。
3. 建立 MAL-only 外部新考卷。

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

### MAL-only dual-target exam

檔案：

- `data/external_transformed/mal_july2025_mal_only_dual_target_exam.csv`

用途：

1. 主外部新資料考卷。
2. 同一批樣本同時評估 popularity 與 score。

目前狀態：

1. rows：2,510。
2. 有 `members`：2,510。
3. 有 `score * 10`：2,510。
4. 有 release year + quarter：2,482。
5. 有 text description：2,510。
6. 可直接跑目前 full multimodal model：0。

### MAL-only popularity-only exam

檔案：

- `data/external_transformed/mal_july2025_mal_only_popularity_exam.csv`

用途：

1. 補充實驗。
2. 只評估 popularity 泛化能力。

目前狀態：

1. rows：9,545。
2. 有 `members`：9,545。
3. 有 release year + quarter：7,035。
4. 有 text description：9,541。
5. 可直接跑目前 full multimodal model：0。

## 4. 為什麼目前不能直接跑 full multimodal

MAL-only 外部資料不是內部 AniList rows，因此缺少目前 full multimodal pipeline 依賴的部分資產。

主要缺口：

1. 沒有現成 text embeddings。
2. 沒有現成 image embeddings。
3. 多數沒有內部 AniList ID。
4. 沒有完整 cover image local asset。
5. RAG 檢索流程目前以 AniList ID 與內部 embedding index 為中心。

因此，目前 MAL-only 考卷已是清洗後 evaluation contract，但還不是可直接餵進 full model 的 inference table。

## 5. 後續 adapter 需求

等模型架構穩定後，應新增 MAL-only inference adapter。

最低需求：

1. 讀取 MAL-only exam CSV。
2. 將 `external_exam_id` 作為外部 row id。
3. 將 MAL metadata 轉成模型 metadata schema。
4. 產生或載入 text embeddings。
5. 定義 image-missing 策略。
6. 定義沒有 AniList ID 時的 RAG fallback。
7. 輸出 prediction 檔，保留 `mal_id`, `external_exam_id`, predictions, external labels。

建議先做 text + metadata smoke test，再接 full multimodal。

## 6. 重跑方式

```bash
python scripts/external/prepare_external_evaluation_assets.py
```

若要將既有 prediction 檔與 MAL aligned labels 比對：

```bash
python scripts/external/evaluate_external_predictions.py \
  --predictions-root ".exp/baseline/results/39/predictions/C2-ProjectInputCTNNDualVisualReconstruction" \
  --split test \
  --output-prefix run39_c2_dual_visual_mal_july2025_external
```

