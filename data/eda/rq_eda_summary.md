# RQ-oriented EDA Summary

- Generated at (UTC): `2026-04-23T08:05:51.662201+00:00`
- Raw rows: `20324`
- Processed rows: `20324`

## Snapshot Control Evidence

- Corr(release_year, popularity_raw): `0.1529367822712231`
- Corr(release_year, popularity_quarter_pct): `-0.046774356538458246`
- Absolute correlation reduction: `0.10616242573276485`

## RQ1 Proxy (Retrieval/Metadata Readiness)

- Studios available ratio: `1.0`
- Genres available ratio: `1.0`
- Studios non-empty ratio: `0.8137669750049203`
- Genres non-empty ratio: `0.8686282227907892`
- Split `test` rows: `9110`
- Split `val` rows: `5440`
- Split `train` rows: `4831`
- Split `holdout_unknown` rows: `943`

### Popularity Bucket Balance by Split
- `test`: {'cold_0_25': 0.24785949506037322, 'hot_50_75': 0.24939626783754115, 'top_75_100': 0.25170142700329307, 'warm_25_50': 0.25104281009879253}
- `train`: {'cold_0_25': 0.23369902711653903, 'hot_50_75': 0.24259987580211137, 'top_75_100': 0.26950941833988823, 'warm_25_50': 0.25419167874146137}
- `val`: {'cold_0_25': 0.24669117647058825, 'hot_50_75': 0.24926470588235294, 'top_75_100': 0.253125, 'warm_25_50': 0.2509191176470588}

## RQ2 Proxy (Multimodal Readiness)

- Text description available ratio: `0.9415961424916355`
- Image cover available ratio: `1.0`
- Trailer id available ratio: `0.3685298169651643`

### Multimodal Coverage by Split
- `holdout_unknown`: {'text_description_available_ratio': 0.9066808059384942, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.2046659597030753}
- `test`: {'text_description_available_ratio': 0.9237102085620198, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.6647639956092206}
- `train`: {'text_description_available_ratio': 0.9507348375077623, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.04160629269302422}
- `val`: {'text_description_available_ratio': 0.9694852941176471, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.19117647058823528}
