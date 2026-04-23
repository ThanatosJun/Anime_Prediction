# Multimodal Input Export Summary

- Generated at (UTC): `2026-04-23T09:38:05.504394+00:00`
- Rows: `20324`
- Columns: `21`

## Feature Contract

- Join key: `id`
- Target columns: `['id', 'release_year', 'release_quarter', 'split_pre_release_effective', 'is_model_split', 'popularity', 'meanScore', 'popularity_quarter_pct', 'popularity_quarter_bucket']`
- Raw multimodal columns: `['id', 'title_romaji', 'title_english', 'description', 'coverImage_medium', 'bannerImage', 'trailer_id', 'trailer_site', 'trailer_thumbnail']`
- Availability flags: `['has_text_description', 'has_cover_image', 'has_banner_image', 'has_trailer']`

## Modality Coverage

- `has_text_description`: `0.9415961424916355`
- `has_cover_image`: `1.0`
- `has_banner_image`: `0.3615922062586105`
- `has_trailer`: `0.36852981696516435`

## Split Counts

- `train`: 13376
- `test`: 3087
- `val`: 2918
- `holdout_unknown`: 943
