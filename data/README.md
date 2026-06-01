# Data Directory

This directory stores the versioned data snapshots and reproducible summaries
used by the project. Large generated assets, downloaded images, and private
external source files should remain outside git.

## Tracked areas

- `raw/`: source manifest and dataset description for the AniList data import.
- `interim/`: intermediate cleaned AniList snapshot.
- `processed/`: train/validation/test/holdout tables used by model pipelines.
- `fussion/`: fusion-branch metadata snapshots, including the v2 split files.
- `eda/`: reproducible EDA summaries and figures.
- `external_transformed/`: compact summaries for external dataset alignment and
  external evaluation assets.
- `fetch_log.csv`: image download log used by the image branch.

## Ignored or local-only areas

- Downloaded image folders such as `data/image/`.
- External raw datasets under `outtestdataset/`.
- Large generated CSV artifacts from `scripts/external/`.

## Rebuild entry points

Run the main data pipeline from the project root:

```bash
python scripts/pipeline/generate_raw_manifest.py
python scripts/pipeline/build_interim_dataset.py
python scripts/pipeline/build_processed_dataset.py
python scripts/pipeline/export_multimodal_inputs.py
```

Run EDA summaries with:

```bash
python scripts/eda/run_baseline_eda.py
python scripts/eda/run_decision_eda.py
python scripts/eda/run_rq_eda.py
python scripts/eda/run_holdout_unknown_diagnostic.py
python scripts/eda/run_column_lineage_report.py
```

Run external-evaluation asset preparation with:

```bash
python scripts/external/prepare_external_evaluation_assets.py
```
