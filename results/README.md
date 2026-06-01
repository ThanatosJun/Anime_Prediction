# Results Directory

This directory contains legacy image-branch model outputs that are currently
tracked for handoff compatibility.

## Current tracked run

- `01/best/`: exported best Swin model files for the image branch.
- `01/checkpoint/`: epoch checkpoints from the same run.
- `01/logs/`: TensorBoard event log.

## Usage

The image branch reads this location through
`src/image_branch/configs/image_process_config.yaml`:

```yaml
output:
  results_dir: results
  run_id: "01"
```

Prediction entry point:

```bash
python -m src.image_branch.run_predict
```

## Version-control note

New training outputs should normally stay out of git unless the team explicitly
decides to preserve a small checkpoint for reproducibility or handoff.
