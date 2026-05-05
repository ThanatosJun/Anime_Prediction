# 缺值狀態（最新快照）

資料來源：
- `data/processed/anilist_anime_data_processed_v1.csv`
- `data/processed/anilist_anime_multimodal_input_v1.csv`

參考腳本：
- `scripts/eda/run_missing_value_report.py`

執行期輸出參考（本機檔案，未納入版控）：
- `data/eda/missing_value_report.json`
- `data/eda/missing_value_report.md`

## processed_v1

- 全部資料筆數：`20324`
- 排除 holdout 後（`is_model_split == True`）筆數：`19381`

### 全資料缺值
- `voice_actor_names`: `8426` (`41.4584%`)
- `startDate_month`: `944` (`4.6448%`)
- `release_date`: `944` (`4.6448%`)
- `release_quarter_key`: `943` (`4.6398%`)
- `release_quarter`: `943` (`4.6398%`)
- `popularity_quarter_pct`: `943` (`4.6398%`)
- `popularity_quarter_bucket`: `943` (`4.6398%`)
- `season`: `943` (`4.6398%`)
- `format`: `1` (`0.0049%`)

### 排除 holdout_unknown 後缺值
- `voice_actor_names`: `7543` (`38.9196%`)
- `startDate_month`: `1` (`0.0052%`)
- `release_date`: `1` (`0.0052%`)

## multimodal_input_v1

- 全部資料筆數：`20324`
- 排除 holdout 後（`is_model_split == True`）筆數：`19381`

### 全資料缺值
- `bannerImage`: `12975` (`63.8408%`)
- `trailer_site`: `12834` (`63.1470%`)
- `trailer_id`: `12834` (`63.1470%`)
- `trailer_thumbnail`: `12834` (`63.1470%`)
- `title_english`: `10623` (`52.2683%`)
- `description`: `1239` (`6.0962%`)
- `release_quarter`: `943` (`4.6398%`)
- `popularity_quarter_bucket`: `943` (`4.6398%`)
- `popularity_quarter_pct`: `943` (`4.6398%`)

### 排除 holdout_unknown 後缺值
- `bannerImage`: `12145` (`62.6645%`)
- `trailer_site`: `12084` (`62.3497%`)
- `trailer_id`: `12084` (`62.3497%`)
- `trailer_thumbnail`: `12084` (`62.3497%`)
- `title_english`: `9947` (`51.3235%`)
- `description`: `1150` (`5.9336%`)
