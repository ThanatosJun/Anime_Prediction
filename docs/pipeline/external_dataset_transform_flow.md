# 外部資料轉換流程（External Dataset Transform Flow）

本指南說明如何把新的外部動畫資料集轉成專案使用的 processed 與 multimodal 契約格式。

## 1) 目標

當組員拿到新的外部動畫 CSV 時，需要可重現的流程輸出：
- 適用 tabular / fusion baseline 的 processed 風格契約
- 適用 text / image / trailer 分支的 multimodal 風格契約

轉換腳本會沿用目前專案規範，並將輸出寫入：
- `data/external_transformed/`

## 2) 必要輸入

- 外部來源 CSV（欄位可不同），例如：
  - `data/external/new_snapshot.csv`
- 欄位映射 JSON（`external_col -> project_col`），例如：
  - `docs/pipeline/external_schema_mapping_example.json`

## 3) 執行指令

於專案根目錄執行：

```bash
python scripts/external/transform_external_dataset.py \
  --input-csv data/external/new_snapshot.csv \
  --mapping-json docs/pipeline/external_schema_mapping_example.json \
  --output-prefix external_v1
```

## 4) 腳本會做什麼

1. **欄位映射（Schema mapping）**
   - 透過 mapping JSON，把外部欄位改名為專案欄位。
2. **Interim 風格規則**
   - 套用現有補值策略：
     - `episodes`: format 中位數 -> 全域中位數
     - `duration`: format 中位數 -> 全域中位數
     - `averageScore`: `meanScore` 回補 -> 全域中位數
     - `seasonYear`: `startDate_year` 回補
     - `title_english`: `title_romaji` 回補
     - `voice_actor_names`: 空字串回補
3. **Processed 風格規則**
   - 關鍵數值欄位非負裁切
   - 依目前專案百分位設定進行 clipping
   - 衍生欄位：
     - `release_year`
     - `release_quarter`
     - `release_quarter_key`
     - `popularity_quarter_pct`
     - `popularity_quarter_bucket`
   - 標記為推論專用 split：
     - `split_pre_release_effective = inference_only`
     - `is_model_split = False`
4. **Multimodal 契約輸出**
   - 新增旗標欄位：
     - `has_text_description`
     - `has_cover_image`
     - `has_banner_image`
     - `has_trailer`

## 5) 輸出檔案

會產生帶時間戳的檔案：

- `data/external_transformed/<prefix>_processed_contract_<ts>.csv`
- `data/external_transformed/<prefix>_multimodal_contract_<ts>.csv`
- `data/external_transformed/<prefix>_transform_summary_<ts>.json`

## 6) Mapping JSON 範本

請建立 `docs/pipeline/external_schema_mapping_example.json`，內容為簡單字典：

```json
{
  "anime_id": "id",
  "title": "title_romaji",
  "english_title": "title_english",
  "season_year": "seasonYear",
  "season_name": "season",
  "score_mean": "meanScore",
  "score_avg": "averageScore",
  "cover_url": "coverImage_medium",
  "synopsis": "description",
  "cast_names": "voice_actor_names"
}
```

## 7) 團隊備註

- 此流程用途是 **外部資料轉換 / 推論前準備**，不是訓練 split 產生流程。
- 若要把外部資料併入訓練實驗，仍需走內部 pipeline 審核與 split 政策討論。
- 建議依資料供應商或資料版本管理 mapping 檔案，確保未來更新可重現。
