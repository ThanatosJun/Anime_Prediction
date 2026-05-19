import torch
import pandas as pd

from src.model import get_embedding
from util.image_process import get_transform_original
from util.dataset import AnimeImageDataset, get_dataloader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def predict_one_col(model, loader, device) -> dict:
    model.eval()
    results = {}
    with torch.no_grad():
        for orig, _, idxs in loader:
            if isinstance(orig, torch.Tensor):
                # 一般路徑：(B, 3, 224, 224)
                orig = orig.to(device)
                embs = get_embedding(model, orig)          # (B, 1024)
            else:
                # YOLO 路徑：List[Tensor(N_i, 3, 224, 224)]
                batch_embs = []
                for crops in orig:
                    crops = crops.to(device)
                    emb = get_embedding(model, crops)      # (N_i, 1024)
                    batch_embs.append(emb.mean(dim=0))     # (1024,)
                embs = torch.stack(batch_embs)             # (B, 1024)
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


def predict(model, config: dict, device, use_yolo:bool) -> None:
    image_dir  = config['data']['image_dir']
    test_df    = pd.read_csv(config['data']['split_csv']['test'])
    transform  = get_transform_original(config['data']['image_size'])
    batch_size = config['training']['batch_size']

    cover_loader = get_dataloader(
        AnimeImageDataset(test_df, image_dir, 'coverImage_medium', transform, transform, use_yolo=use_yolo),
        batch_size, shuffle=False, use_yolo=use_yolo,
    )
    # banner不要yolo
    banner_loader = get_dataloader(
        AnimeImageDataset(test_df, image_dir, 'bannerImage', transform, transform, use_yolo=False),
        batch_size, shuffle=False, use_yolo=False,
    )

    cover_embs  = predict_one_col(model, cover_loader, device)
    banner_embs = predict_one_col(model, banner_loader, device)

    df = merge_embeddings(cover_embs, banner_embs)
    save_embeddings(df, config['output']['embedding_path'])
