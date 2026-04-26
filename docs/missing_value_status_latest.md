# Missing Value Status (Latest Snapshot)

Generated from:
- `data/processed/anilist_anime_data_processed_v1.csv`
- `data/processed/anilist_anime_multimodal_input_v1.csv`

Reference script:
- `scripts/run_missing_value_report.py`

Reference runtime outputs (local, ignored by git):
- `data/eda/missing_value_report.json`
- `data/eda/missing_value_report.md`

## processed_v1

- Full rows: `20324`
- Without holdout (`is_model_split == True`) rows: `19381`

### Full data (missing)
- `voice_actor_names`: `8426` (`41.4584%`)
- `source`: `2320` (`11.4151%`)
- `release_date`: `1406` (`6.9179%`)
- `startDate_day`: `1405` (`6.9130%`)
- `startDate_month`: `944` (`4.6448%`)
- `release_quarter_key`: `943` (`4.6398%`)
- `release_quarter`: `943` (`4.6398%`)
- `popularity_quarter_pct`: `943` (`4.6398%`)
- `popularity_quarter_bucket`: `943` (`4.6398%`)
- `season`: `943` (`4.6398%`)
- `title_native`: `167` (`0.8217%`)
- `format`: `1` (`0.0049%`)

### Without holdout_unknown (missing)
- `voice_actor_names`: `7543` (`38.9196%`)
- `source`: `2102` (`10.8457%`)
- `release_date`: `463` (`2.3889%`)
- `startDate_day`: `463` (`2.3889%`)
- `title_native`: `146` (`0.7533%`)
- `startDate_month`: `1` (`0.0052%`)

## multimodal_input_v1

- Full rows: `20324`
- Without holdout (`is_model_split == True`) rows: `19381`

### Full data (missing)
- `bannerImage`: `12975` (`63.8408%`)
- `trailer_site`: `12834` (`63.1470%`)
- `trailer_id`: `12834` (`63.1470%`)
- `trailer_thumbnail`: `12834` (`63.1470%`)
- `title_english`: `10623` (`52.2683%`)
- `description`: `1239` (`6.0962%`)
- `release_quarter`: `943` (`4.6398%`)
- `popularity_quarter_bucket`: `943` (`4.6398%`)
- `popularity_quarter_pct`: `943` (`4.6398%`)

### Without holdout_unknown (missing)
- `bannerImage`: `12145` (`62.6645%`)
- `trailer_site`: `12084` (`62.3497%`)
- `trailer_id`: `12084` (`62.3497%`)
- `trailer_thumbnail`: `12084` (`62.3497%`)
- `title_english`: `9947` (`51.3235%`)
- `description`: `1150` (`5.9336%`)
