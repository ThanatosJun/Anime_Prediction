import torch
import pandas as pd

from src.model import get_embedding
from util.image_process import get_transform_original
from util.dataset import AnimeImageDataset, get_dataloader


def predict_one_col(model, loader, device) -> dict:
    model.eval()
    results = {}
    with torch.no_grad():
        for orig, _, idxs in loader:
            orig = orig.to(device)
            embs = get_embedding(model, orig)
            for i, idx in enumerate(idxs):
                results[int(idx)] = embs[i].cpu().numpy()
    return results


def merge_embeddings(cover_embs: dict, banner_embs: dict) -> pd.DataFrame:
    all_idx = sorted(set(cover_embs.keys()) | set(banner_embs.keys()))
    rows = [
        {
            'idx':            idx,
            'coverImage_emb': cover_embs.get(idx, None),
            'bannerImage_emb': banner_embs.get(idx, None),
        }
        for idx in all_idx
    ]
    return pd.DataFrame(rows)


def save_embeddings(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def predict(model, config: dict, device) -> None:
    image_dir  = 'data/image'
    test_df    = pd.read_csv('data/processed/anilist_anime_multimodal_input_test.csv')
    transform  = get_transform_original(config['data']['image_size'])
    batch_size = config['training']['batch_size']

    cover_loader = get_dataloader(
        AnimeImageDataset(test_df, image_dir, 'coverImage_medium', transform, transform),
        batch_size, shuffle=False,
    )
    banner_loader = get_dataloader(
        AnimeImageDataset(test_df, image_dir, 'bannerImage', transform, transform),
        batch_size, shuffle=False,
    )

    cover_embs  = predict_one_col(model, cover_loader, device)
    banner_embs = predict_one_col(model, banner_loader, device)

    df = merge_embeddings(cover_embs, banner_embs)
    save_embeddings(df, config['output']['embedding_path'])
