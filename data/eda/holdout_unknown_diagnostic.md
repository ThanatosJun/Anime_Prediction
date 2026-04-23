# Holdout Unknown Diagnostic

- Generated at (UTC): `2026-04-23T16:32:24.298644+00:00`
- Total rows: `20324`
- holdout_unknown rows: `943` (4.64%)
- Policy: `holdout_unknown samples are excluded from model train/val/test.`

## Effective Split Counts

- `train`: 13376
- `test`: 3087
- `val`: 2918
- `holdout_unknown`: 943

## Temporal Missing Focus (holdout_unknown only)

- `release_year` missing ratio: `0.00%`
- `release_quarter` missing ratio: `100.00%`
- `release_quarter_key` missing ratio: `100.00%`
- `seasonYear` missing ratio: `0.00%`
- `startDate_year` missing ratio: `0.00%`
- `startDate_month` missing ratio: `100.00%`

## Distribution Gap vs Model Splits

- `popularity`: mean_gap=-12526.596595095405, median_gap=-992.0
- `averageScore`: mean_gap=-15.12834436849112, median_gap=-17.0
- `episodes`: mean_gap=-7.268739710366708, median_gap=-1.0
- `duration`: mean_gap=-10.210391275950368, median_gap=-16.0
- `favourites`: mean_gap=-284.60649233763786, median_gap=-11.0
- `trending`: mean_gap=-0.09712308569526966, median_gap=0.0
