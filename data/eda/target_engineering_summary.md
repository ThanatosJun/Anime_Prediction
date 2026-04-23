# Target Engineering Summary

- Rule version: `decision_eda_v3`
- Input file: `data/interim/anilist_anime_data_interim_20260424.csv`

## Popularity Quarter Target

- `top_75_100`: 4972
- `warm_25_50`: 4880
- `hot_50_75`: 4800
- `cold_0_25`: 4729
- `nan`: 943

## Temporal Pre-release Split

- Train quarters: `269`
- Val quarters: `14`
- Test quarters: `16`
- `train` rows: 13376
- `test` rows: 3087
- `val` rows: 2918
- `unknown` rows: 943

## Unknown Split Policy

- Policy: `exclude_unknown_from_model_splits`
- Excluded unknown rows from model splits: `943`
- Effective `train` rows: 13376
- Effective `test` rows: 3087
- Effective `val` rows: 2918
- Effective `holdout_unknown` rows: 943
