import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd

from src.model import get_embedding, get_stage_embeddings
from util.image_process import get_transform_original
from util.dataset import AnimeImageDataset, get_dataloader


def predict_one_col(model, loader, device, use_stage: bool = False) -> dict:
    """
    回傳 dict:
      use_stage=False: {idx: Tensor(1024,)}
      use_stage=True:  {idx: [Tensor(128,), Tensor(256,), Tensor(512,), Tensor(1024,)]}
    """
    model.eval()
    results = {}
    with torch.no_grad():
        for orig, _, idxs in loader:
            if isinstance(orig, torch.Tensor):
                orig = orig.to(device)
                if use_stage:
                    # [(B,128), (B,256), (B,512), (B,1024)]
                    stage_embs = get_stage_embeddings(model, orig)
                else:
                    embs = get_embedding(model, orig)  # (B, 1024)
            else:
                # YOLO 路徑：List[Tensor(N_i, 3, 224, 224)]
                if use_stage:
                    # 每個 sample 各 stage mean pool → [(128,),(256,),(512,),(1024,)]
                    # 再 stack 成 [(B,128),(B,256),(B,512),(B,1024)]
                    per_sample = []
                    for crops in orig:
                        crops = crops.to(device)
                        s = get_stage_embeddings(model, crops)  # [(N_i,C), ...]
                        per_sample.append([e.mean(dim=0) for e in s])  # [(C,), ...]
                    stage_embs = [
                        torch.stack([per_sample[b][s] for b in range(len(per_sample))])
                        for s in range(4)
                    ]  # [(B,128),(B,256),(B,512),(B,1024)]
                else:
                    per_sample = []
                    for crops in orig:
                        crops = crops.to(device)
                        emb = get_embedding(model, crops)  # (N_i, 1024)
                        per_sample.append(emb.mean(dim=0))  # (1024,)
                    embs = torch.stack(per_sample)  # (B, 1024)

            for i, idx in enumerate(idxs):
                if use_stage:
                    results[int(idx)] = [s[i].cpu() for s in stage_embs]
                else:
                    results[int(idx)] = embs[i].cpu()
    return results


def merge_embeddings(cover_embs: dict, banner_embs: dict, use_stage: bool = False) -> pd.DataFrame:
    all_idx = sorted(set(cover_embs.keys()) | set(banner_embs.keys()))
    rows = []
    for idx in all_idx:
        cover_val  = cover_embs.get(idx, None)
        banner_val = banner_embs.get(idx, None)
        if use_stage:
            # 各 stage 分欄：coverImage_emb_s0~s3、bannerImage_emb_s0~s3
            row = {'idx': idx}
            for s in range(4):
                row[f'coverImage_emb_s{s}']  = cover_val[s].numpy()  if cover_val  is not None else None
                row[f'bannerImage_emb_s{s}'] = banner_val[s].numpy() if banner_val is not None else None
        else:
            row = {
                'idx':             idx,
                'coverImage_emb':  cover_val.numpy()  if cover_val  is not None else None,
                'bannerImage_emb': banner_val.numpy() if banner_val is not None else None,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def save_embeddings(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def predict(model, config: dict, device, use_yolo: bool) -> None:
    image_dir  = config['data']['image_dir']
    test_df    = pd.read_csv(config['data']['split_csv']['test'])
    transform  = get_transform_original(config['data']['image_size'])
    batch_size = config['training']['batch_size']
    use_stage  = config['model'].get('stage', False)

    cover_loader = get_dataloader(
        AnimeImageDataset(test_df, image_dir, 'coverImage_medium', transform, transform, use_yolo=use_yolo),
        batch_size, shuffle=False, use_yolo=use_yolo,
    )
    banner_loader = get_dataloader(
        AnimeImageDataset(test_df, image_dir, 'bannerImage', transform, transform, use_yolo=False),
        batch_size, shuffle=False, use_yolo=False,
    )

    cover_embs  = predict_one_col(model, cover_loader,  device, use_stage=use_stage)
    banner_embs = predict_one_col(model, banner_loader, device, use_stage=use_stage)

    df = merge_embeddings(cover_embs, banner_embs, use_stage=use_stage)
    save_embeddings(df, config['output']['embedding_path'])
