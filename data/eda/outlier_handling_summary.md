# Outlier Handling Summary

- Input file: `data/interim/anilist_anime_data_interim_20260423.csv`
- Output rows: `20324`
- Output columns: `25`

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
- `averageScore`: [30.0000, 83.0000], clipped=325
- `meanScore`: [30.0000, 83.0000], clipped=388
- `popularity`: [25.0000, 231528.9000], clipped=386
- `favourites`: [0.0000, 7852.5500], clipped=204
- `trending`: [0.0000, 6.0000], clipped=190
