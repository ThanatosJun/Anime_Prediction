import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.image_branch.model import get_embedding
from src.image_branch.image_process import get_transform_original
from src.image_branch.dataset import AnimeImageDataset, get_dataloader


def _infer_col(model, df: pd.DataFrame, image_dir: str,
               image_col: str, transform, batch_size: int, device) -> dict:
    """Returns {anime_id: np.array(1024)} for all rows in df."""
    loader = get_dataloader(
        AnimeImageDataset(df, image_dir, image_col, transform, transform),
        batch_size, shuffle=False,
    )
    model.eval()
    results = {}
    with torch.no_grad():
        for orig, _, idxs in loader:
            orig = orig.to(device)
            embs = get_embedding(model, orig)          # (B, 1024)
            for i, idx in enumerate(idxs):
                results[int(idx)] = embs[i].cpu().numpy()
    return results


def predict(model, config: dict, device) -> None:
    """
    Run coverImage_medium inference on ALL anime (train + val + test),
    output data/processed/image_embeddings.parquet with columns:
        id, img_0, img_1, ..., img_1023
    """
    image_dir  = config['data']['image_dir']
    image_size = config['data']['image_size']
    batch_size = config['training']['batch_size']
    out_path   = config['output']['embedding_path']
    transform  = get_transform_original(image_size)

    # load full dataset (all splits)
    full_df = pd.read_csv(config['data']['csv_path'])
    print(f"Running inference on {len(full_df)} anime (coverImage_medium)…")

    embs = _infer_col(model, full_df, image_dir, 'coverImage_medium',
                      transform, batch_size, device)

    # build flat parquet: id + img_0..img_767
    ids = sorted(embs.keys())
    mat = np.stack([embs[i] for i in ids], axis=0)   # (N, 1024)
    img_cols = [f"img_{j}" for j in range(mat.shape[1])]
    out_df = pd.DataFrame(mat, columns=img_cols)
    out_df.insert(0, "id", ids)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} embeddings → {out_path}  shape={out_df.shape}")
