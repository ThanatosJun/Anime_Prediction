# Legacy Utility Modules

This directory contains earlier shared helpers for image loading, dataset
wrapping, prediction, and training.

## Contents

- `dataset.py`: legacy image dataset and dataloader helpers.
- `getImage.py`: earlier image downloader.
- `image_process.py`: image loading and resizing helpers.
- `predictor.py`: legacy image embedding export helper.
- `train.py`: earlier image training loop.
- `split_images_by_split.py`: utility for arranging downloaded images by split.
- `test_image_embedding.py`: quick local check for image embedding behavior.

## Current status

The maintained package entry point is `src/image_branch/`. Keep this directory
for compatibility with older notebooks or handoff scripts, but prefer
`src.image_branch` for new work.
