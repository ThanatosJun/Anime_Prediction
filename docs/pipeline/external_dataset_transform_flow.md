# External Dataset Transform Flow

This guide describes how to transform a new external dataset into the project's processed and multimodal contracts.

## 1) Goal

When a teammate receives a new external anime CSV, we want a reproducible way to output:
- processed-style contract for tabular/fusion baseline
- multimodal-style contract for text/image/trailer branches

The transformer keeps current project conventions and writes outputs under:
- `data/external_transformed/`

## 2) Required Inputs

- External source CSV (any schema), for example:
  - `data/external/new_snapshot.csv`
- Column mapping JSON (`external_col -> project_col`), for example:
  - `docs/external_schema_mapping_example.json`

## 3) Run Command

From repository root:

```bash
python scripts/transform_external_dataset.py \
  --input-csv data/external/new_snapshot.csv \
  --mapping-json docs/external_schema_mapping_example.json \
  --output-prefix external_v1
```

## 4) What the Script Does

1. **Schema mapping**
   - Renames external columns to project columns using mapping JSON.
2. **Interim-like rules**
   - Applies existing-style imputation:
     - `episodes`: format median -> global median
     - `duration`: format median -> global median
     - `averageScore`: `meanScore` fallback -> global median
     - `seasonYear`: `startDate_year` fallback
     - `title_english`: `title_romaji` fallback
     - `voice_actor_names`: empty string fallback
3. **Processed-like rules**
   - Non-negative clipping for key numerics
   - Percentile clipping with current project quantiles
   - Derives:
     - `release_year`
     - `release_quarter`
     - `release_quarter_key`
     - `popularity_quarter_pct`
     - `popularity_quarter_bucket`
   - Marks split as inference-only:
     - `split_pre_release_effective = inference_only`
     - `is_model_split = False`
4. **Multimodal contract export**
   - Adds:
     - `has_text_description`
     - `has_cover_image`
     - `has_banner_image`
     - `has_trailer`

## 5) Outputs

Timestamped files are generated:

- `data/external_transformed/<prefix>_processed_contract_<ts>.csv`
- `data/external_transformed/<prefix>_multimodal_contract_<ts>.csv`
- `data/external_transformed/<prefix>_transform_summary_<ts>.json`

## 6) Mapping JSON Template

Create `docs/external_schema_mapping_example.json` with a simple dictionary:

```json
{
  "anime_id": "id",
  "title": "title_romaji",
  "english_title": "title_english",
  "season_year": "seasonYear",
  "season_name": "season",
  "score_mean": "meanScore",
  "score_avg": "averageScore",
  "cover_url": "coverImage_medium",
  "synopsis": "description",
  "cast_names": "voice_actor_names"
}
```

## 7) Team Notes

- This flow is for **external transformation/inference preparation**, not training split generation.
- If external data should be merged into training experiments, follow with the normal internal pipeline review and split policy discussions.
- Keep mapping files versioned per source vendor/dataset so future refreshes remain reproducible.
