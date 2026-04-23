# Column Lineage Summary

- Generated at (UTC): `2026-04-23T08:53:28.619497+00:00`
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
