# Image Branch

This package downloads anime cover/banner images, fine-tunes a Swin image
encoder, and exports image embeddings for downstream multimodal fusion.

## Main entry points

Run from the project root:

```bash
python -m src.image_branch.run_fetch
python -m src.image_branch.run_train
python -m src.image_branch.run_predict
```

For the full legacy sequence:

```bash
python -m src.image_branch.run_main
```

## Important files

- `configs/image_process_config.yaml`: data paths, image columns, model name,
  training hyperparameters, and output paths.
- `get_image.py`: downloads images listed in the processed AniList table.
- `dataset.py`: builds image pairs for contrastive training.
- `model.py`: loads the Swin encoder.
- `train.py`: fine-tunes the image encoder.
- `predictor.py`: exports embeddings for downstream use.
- `summarize_fetch_coverage.py`: summarizes image download coverage.

## Inputs and outputs

Default inputs:

- `data/processed/anilist_anime_multimodal_input_v1.csv`
- `data/processed/anilist_anime_multimodal_input_{train,val,test}.csv`

Default local image directory:

- `data/image/`

Default tracked handoff checkpoint:

- `results/01/best/`

Default embedding output:

- `.exp/image_embedding/image_embeddings.parquet`

Downloaded images and generated embeddings are local artifacts and should not be
committed by default.
