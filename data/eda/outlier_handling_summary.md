# Outlier Handling Summary

- Rule version: `decision_eda_v3`
- Input file: `data/interim/anilist_anime_data_interim_20260426.csv`
- Output rows: `20324`
- Output columns: `42`

## Non-negative Enforcement

- `episodes`: negative_to_zero=0
- `duration`: negative_to_zero=0
- `averageScore`: negative_to_zero=0
- `meanScore`: negative_to_zero=0
- `popularity`: negative_to_zero=0
- `favourites`: negative_to_zero=0
- `trending`: negative_to_zero=0

## Percentile Clipping

- `episodes`: [1.0000, 104.0000], clipped=190
- `duration`: [1.0000, 115.0000], clipped=187
- `averageScore`: [28.0000, 85.0000], clipped=177
- `meanScore`: [27.0000, 85.0000], clipped=173
- `popularity`: [25.0000, 231528.9000], clipped=386
- `favourites`: [0.0000, 7852.5500], clipped=204
- `trending`: [0.0000, 1.0000], clipped=831
