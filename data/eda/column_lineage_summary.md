# Column Lineage Summary

- Generated at (UTC): `2026-04-23T08:57:29.382461+00:00`
- Column counts: raw=`61`, interim=`25`, processed=`33`

## Raw -> Interim

- Kept columns: `24`
- Dropped columns: `37`
- Added columns: `1`

### Added in Interim
- `release_date`: derived in interim (`add_release_date`)

## Interim -> Processed

- Kept columns: `25`
- Dropped columns: `0`
- Added columns: `8`

### Added in Processed
- `is_model_split`: derived in processed (`_apply_unknown_split_policy`)
- `popularity_quarter_bucket`: derived in processed (`_add_popularity_quarter_target`)
- `popularity_quarter_pct`: derived in processed (`_add_popularity_quarter_target`)
- `release_quarter`: derived in processed (`_derive_release_quarter`)
- `release_quarter_key`: derived in processed (`_derive_release_quarter`)
- `release_year`: derived in processed (`_derive_release_quarter`)
- `split_pre_release`: derived in processed (`_apply_pre_release_temporal_split`)
- `split_pre_release_effective`: derived in processed (`_apply_unknown_split_policy`)

## Raw -> Processed Direct View

- Dropped from raw by final stage: `37`
- Added by final stage: `9`

## Raw -> Interim Drop Reasons

- `airingSchedule`: nested schedule nodes; not required for current pre-release baseline target engineering
- `bannerImage`: auxiliary media field; baseline uses tabular contract and defers image handling to dedicated stage
- `chapters`: manga-oriented field; 100% missing for this anime-focused pipeline
- `characters`: high-cardinality nested structure; deferred to dedicated retrieval/graph feature stage
- `coverImage_color`: auxiliary color metadata; low impact baseline feature and missingness present
- `coverImage_extraLarge`: raw media URL variant; excluded from baseline tabular contract
- `coverImage_large`: raw media URL variant; excluded from baseline tabular contract
- `coverImage_medium`: tracked in RQ coverage analysis but not used as direct tabular scalar feature
- `description`: kept for RQ readiness analysis but excluded from baseline tabular feature set (reserved for text encoder stage)
- `endDate_day`: post-release endpoint detail not required for pre-release feature contract
- `endDate_month`: post-release endpoint detail not required for pre-release feature contract
- `endDate_year`: post-release endpoint detail not required for pre-release feature contract
- `externalLinks`: nested external metadata; excluded to keep baseline feature contract compact
- `hashtag`: high missingness and unstable social metadata
- `idMal`: external ID mapping field; excluded from baseline predictive feature set
- `isFavourite`: user-level interaction flag; unstable and potentially leakage-prone
- `isLicensed`: platform/legal flag not required in current baseline contract
- `isLocked`: platform operational flag; not semantically meaningful for target task
- `nextAiringEpisode`: post-schedule dynamic field; excluded to avoid temporal inconsistency/leakage risk
- `rankings`: nested ranking history; may introduce post-release leakage and high variance
- `recommendations`: nested recommendation graph payload; excluded from baseline tabular preprocessing
- `relations`: graph-style relation payload; handled in retrieval augmentation stage instead of baseline tabular stage
- `reviews`: nested text payload with variable quality/length; deferred to dedicated NLP stage
- `seasonInt`: redundant encoded season field; season + seasonYear retained
- `siteUrl`: identifier-style URL field; non-semantic for baseline model input
- `staff`: high-cardinality nested structure; deferred to dedicated retrieval/graph feature stage
- `stats_scoreDistribution`: aggregated platform distribution payload; excluded from row-level baseline contract
- `stats_statusDistribution`: aggregated platform distribution payload; excluded from row-level baseline contract
- `streamingEpisodes`: nested streaming payload; not stable for pre-release baseline feature contract
- `synonyms`: high-variance synonym list; deferred to dedicated text normalization stage
- `tags`: high-cardinality nested tags; deferred to dedicated text/tag embedding pipeline
- `title_userPreferred`: redundant with retained title fields and may vary by locale/user preference
- `trailer_id`: retained as availability metric in EDA but excluded from baseline tabular contract to avoid sparse-noisy key field
- `trailer_site`: sparse categorical trailer metadata; deferred to multimodal source availability analysis
- `trailer_thumbnail`: media URL field; excluded from baseline tabular contract
- `updatedAt`: platform update timestamp; can encode non-stationary platform behavior
- `volumes`: manga-oriented field; 100% missing for this anime-focused pipeline
