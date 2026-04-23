# RQ-oriented EDA Summary

- Generated at (UTC): `2026-04-23T09:38:07.773428+00:00`
- Raw rows: `20324`
- Processed rows: `20324`

## Snapshot Control Evidence

- Corr(release_year, popularity_raw): `0.1529367822712231`
- Corr(release_year, popularity_quarter_pct): `-0.046774356538458246`
- Absolute correlation reduction: `0.10616242573276485`

## Statistical Test Layer

- Bucket balance permutation test (max TVD): stat=`0.005934000000495085`, p=`0.9925187032418953`
- Snapshot reduction bootstrap CI95: [`0.0885708825502303`, `0.12215550704813904`], mean=`0.1064216371334734`
- Coverage permutation `text_available`: stat=`0.05221452763084111`, p=`0.0024937655860349127`
- Coverage permutation `cover_available`: stat=`0.0`, p=`1.0`
- Coverage permutation `banner_available`: stat=`0.19077861057091605`, p=`0.0024937655860349127`
- Coverage permutation `trailer_available`: stat=`0.5551406538919965`, p=`0.0024937655860349127`

## RQ1 Proxy (Retrieval/Metadata Readiness)

- Studios available ratio: `1.0`
- Genres available ratio: `1.0`
- Studios non-empty ratio: `0.8137669750049203`
- Genres non-empty ratio: `0.8686282227907892`
- Split `train` rows: `13376`
- Split `test` rows: `3087`
- Split `val` rows: `2918`
- Split `holdout_unknown` rows: `943`

### Popularity Bucket Balance by Split
- `test`: {'cold_0_25': 0.24813735017816652, 'hot_50_75': 0.2491091674765144, 'top_75_100': 0.2520246193715581, 'warm_25_50': 0.25072886297376096}
- `train`: {'cold_0_25': 0.24222488038277512, 'hot_50_75': 0.24686004784688995, 'top_75_100': 0.258747009569378, 'warm_25_50': 0.25216806220095694}
- `val`: {'cold_0_25': 0.24777244688142563, 'hot_50_75': 0.24982864976010966, 'top_75_100': 0.25119945167923236, 'warm_25_50': 0.25119945167923236}

## RQ2 Proxy (Multimodal Readiness)

- Text description available ratio: `0.9415961424916355`
- Image cover available ratio: `1.0`
- Trailer id available ratio: `0.3685298169651643`

### Multimodal Coverage by Split
- `holdout_unknown`: {'text_description_available_ratio': 0.9066808059384942, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.2046659597030753}
- `test`: {'text_description_available_ratio': 0.9115646258503401, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.7719468739876904}
- `train`: {'text_description_available_ratio': 0.9586572966507177, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.21680622009569378}
- `val`: {'text_description_available_ratio': 0.9064427690198766, 'image_cover_available_ratio': 1.0, 'trailer_id_available_ratio': 0.6901987662782728}
