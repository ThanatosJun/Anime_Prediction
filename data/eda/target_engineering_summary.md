# Target Engineering Summary

- Rule version: `decision_eda_v3`
- Input file: `data/interim/anilist_anime_data_interim_20260423.csv`

## Popularity Quarter Target

- `top_75_100`: 5208
- `warm_25_50`: 5115
- `hot_50_75`: 5036
- `cold_0_25`: 4965

## Temporal Pre-release Split

- Train quarters: `209`
- Val quarters: `44`
- Test quarters: `46`
- `test` rows: 9110
- `val` rows: 5440
- `train` rows: 4831
- `unknown` rows: 943

## Unknown Split Policy

- Policy: `exclude_unknown_from_model_splits`
- Excluded unknown rows from model splits: `943`
- Effective `test` rows: 9110
- Effective `val` rows: 5440
- Effective `train` rows: 4831
- Effective `holdout_unknown` rows: 943
