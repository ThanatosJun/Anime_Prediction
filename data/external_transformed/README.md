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
- `mal2025_image_external_eval_contract.csv`
  - MyAnimeList 2025 rows aligned to internal AniList rows with cover image
    URLs, for image-aware external-label sanity checks.
- `mal2025_image_mal_only_dual_target_exam.csv`
  - Conservative MAL-only full multimodal exam rows with cover URL, text,
    metadata, `members`, and `score * 10`.
- `mal2025_image_mal_only_popularity_exam.csv`
  - Conservative MAL-only full multimodal exam rows with cover URL, text,
    metadata, and `members`.

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
- MyAnimeList 2025 image-ready exams:
  - 19,931 source rows
  - 19,544 rows with cover image URLs
  - 19,283 rows with cover URL, text, and release year/quarter
  - 15,485 aligned internal rows with image-ready inputs
  - 3,798 conservative MAL-only popularity rows with image/text/metadata
  - 1,209 conservative MAL-only dual-target rows with image/text/metadata

`mal_july2025_*` remains useful for label sanity checks, but the July 2025
source file has no image column. Use `mal2025_image_*` for full multimodal
external exams.

## External label mapping

- `external_popularity_members`: MAL `members`, used as the external popularity
  count proxy.
- `external_score_0_100`: MAL `score * 10`, roughly aligned to AniList's
  0-100 score scale.
- `external_popularity_rank`: MAL popularity rank. Lower means more popular, so
  keep it for ranking diagnostics only.

## Image asset convention

The image-ready MAL 2025 exams include:

- `external_cover_image_url`: source cover URL from MAL.
- `external_cover_image_path`: intended local download target under
  `data/external_assets/mal2025_image/cover/`.

Generated external image assets are ignored by git.

Use this helper when network access is available:

```bash
python scripts/external/download_external_images.py \
  --exam-csv data/external_transformed/mal2025_image_mal_only_dual_target_exam.csv
```

Then materialize local-ready exam CSVs:

```bash
python scripts/external/prepare_external_local_ready_exams.py
```

This writes ignored CSV artifacts:

- `mal2025_image_mal_only_popularity_exam_local_ready.csv`
- `mal2025_image_mal_only_popularity_exam_missing_local_images.csv`
- `mal2025_image_mal_only_dual_target_exam_local_ready.csv`
- `mal2025_image_mal_only_dual_target_exam_missing_local_images.csv`

It also writes `mal2025_image_local_ready_summary.json` with the row counts for
the current machine.

## Reproducible workflow

From a clean checkout with the external datasets under `outtestdataset/`:

```bash
python scripts/external/prepare_external_evaluation_assets.py
python scripts/external/download_external_images.py \
  --exam-csv data/external_transformed/mal2025_image_mal_only_popularity_exam.csv \
  --sleep 0
python scripts/external/prepare_external_local_ready_exams.py
```

The popularity image exam contains the dual-target rows, so downloading the
popularity exam first also prepares the dual-target exam images.

## Evaluation helper

Existing prediction files can be compared against the aligned external labels:

```bash
python scripts/external/evaluate_external_predictions.py \
  --predictions-root ".exp/baseline/results/39/predictions/C2-ProjectInputCTNNDualVisualReconstruction" \
  --split test \
  --output-prefix run39_c2_dual_visual_mal_july2025_external
```
