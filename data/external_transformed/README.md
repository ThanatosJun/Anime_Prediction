# External Transformed Assets

This directory stores reproducible summaries for external dataset alignment and
evaluation. Large generated CSV files are ignored by git and should be
regenerated locally.

## Regenerate assets

Run from the project root:

```bash
python scripts/external/prepare_external_evaluation_assets.py
```

This expects the external datasets under `outtestdataset/`.

## Generated CSV outputs

The script writes these ignored CSV artifacts:

- `aodb_id_crosswalk.csv`
  - Anime Offline Database bridge table for AniList ID and MAL ID.
- `aodb_holdout_unknown_recovered_rows.csv`
  - `holdout_unknown` rows whose release quarter can be recovered with AODB.
- `anilist_anime_multimodal_input_v1_aodb_recovered_future.csv`
  - Future-work multimodal table with AODB recovery applied.
- `mal_july2025_external_eval_contract.csv`
  - MAL July 2025 rows aligned to internal AniList rows for cross-platform
    external-label evaluation.
- `mal_july2025_mal_only_dual_target_exam.csv`
  - MAL-only external exam rows with both `members` and `score * 10`.
- `mal_july2025_mal_only_popularity_exam.csv`
  - MAL-only external exam rows with `members`.

## Current validated counts

As of the latest run:

- AODB crosswalk:
  - 40,515 rows
  - 20,352 unique AniList IDs
  - 29,932 unique MAL IDs
- AODB holdout recovery:
  - 943 original `holdout_unknown` rows
  - 789 recoverable rows
  - 154 rows remain `holdout_unknown`
- MAL July 2025 aligned external evaluation:
  - 28,635 source rows
  - 19,090 rows mapped to internal AniList IDs
  - 15,590 `external_eval_ready` rows
- MAL-only external exams:
  - 9,545 MAL-only rows with `members`
  - 2,510 MAL-only rows with both `members` and `score * 10`
  - 2,482 dual-target rows have release year and quarter
  - 0 rows are currently ready for the existing full multimodal model without
    generating new features/assets

## External label mapping

- `external_popularity_members`: MAL `members`, used as the external popularity
  count proxy.
- `external_score_0_100`: MAL `score * 10`, roughly aligned to AniList's
  0-100 score scale.
- `external_popularity_rank`: MAL popularity rank. Lower means more popular, so
  keep it for ranking diagnostics only.

## Evaluation helper

Existing prediction files can be compared against the aligned external labels:

```bash
python scripts/external/evaluate_external_predictions.py \
  --predictions-root ".exp/baseline/results/39/predictions/C2-ProjectInputCTNNDualVisualReconstruction" \
  --split test \
  --output-prefix run39_c2_dual_visual_mal_july2025_external
```

