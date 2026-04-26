import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.fussion_branch.meta_encoder import MetaEncoder

IMAGE_EMB_DIM = 1024  # Swin-base pooler_output (hidden_size=1024)


class FusionDataset(Dataset):
    """
    Feature layout per sample:
        [text_embedding]    384-dim   all-MiniLM-L6-v2
        [image_embedding]  1024-dim   Swin-base (optional; zeros if parquet missing)
        [meta + rag]        158-dim   MetaEncoder output
        ─────────────────────────────
        total              1566-dim   (542 without image)
    """

    def __init__(
        self,
        split: str,
        encoder: MetaEncoder,
        meta_dir: str = "data/fussion",
        text_emb_dir: str = "artifacts",
        rag_dir: str = "artifacts",
        image_emb_path: Optional[str] = "data/processed/image_embeddings.parquet",
        target_col: str = "popularity",
        log_transform_target: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        meta_df = pd.read_csv(f"{meta_dir}/fusion_meta_{split}.csv")
        rag_df  = pd.read_parquet(f"{rag_dir}/rag_features_{split}.parquet")
        text_df = pd.read_parquet(f"{text_emb_dir}/text_embeddings_{split}.parquet")

        # align on id
        meta_df = meta_df.set_index("id")
        rag_df  = rag_df.set_index("id")
        text_df = text_df.set_index("id")

        common_ids = meta_df.index.intersection(rag_df.index).intersection(text_df.index)

        # image embedding (optional)
        self.use_image = False
        image_df = None
        if image_emb_path and os.path.exists(image_emb_path):
            raw = pd.read_parquet(image_emb_path)
            # filter to current split by id intersection
            raw = raw.set_index("id")
            img_ids = common_ids.intersection(raw.index)
            if len(img_ids) > 0:
                common_ids = img_ids
                image_df = raw.loc[common_ids].reset_index()
                self.use_image = True
                print(f"  [{split}] image embeddings loaded: {len(image_df)} rows")
        if not self.use_image:
            print(f"  [{split}] image embeddings not found — using zeros (dim={IMAGE_EMB_DIM})")

        meta_df = meta_df.loc[common_ids].reset_index()
        rag_df  = rag_df.loc[common_ids].reset_index()
        text_df = text_df.loc[common_ids].reset_index()

        self.ids = meta_df["id"].values
        N = len(self.ids)

        # text embedding (384)
        emb_cols = [c for c in text_df.columns if c.startswith("emb_")]
        self.text_emb = text_df[emb_cols].values.astype(np.float32)   # (N, 384)

        # image embedding (768) — zeros if not available
        if self.use_image:
            img_cols = [c for c in image_df.columns if c != "id"]
            self.image_emb = image_df[img_cols].values.astype(np.float32)  # (N, 768)
        else:
            self.image_emb = np.zeros((N, IMAGE_EMB_DIM), dtype=np.float32)

        # metadata + rag features (158)
        self.meta_feat = encoder.transform(meta_df, rag_df)

        # target
        raw_target = meta_df[target_col].values.astype(np.float32)
        if log_transform_target:
            raw_target = np.log1p(raw_target)
        self.target = (raw_target - target_mean) / target_std

    @property
    def feature_dim(self) -> int:
        return self.text_emb.shape[1] + self.image_emb.shape[1] + self.meta_feat.shape[1]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        features = np.concatenate([
            self.text_emb[idx],    # 384
            self.image_emb[idx],   # 768
            self.meta_feat[idx],   # 158
        ])
        return {
            "features": torch.from_numpy(features),
            "target":   torch.tensor(self.target[idx], dtype=torch.float32),
            "id":       self.ids[idx],
        }
