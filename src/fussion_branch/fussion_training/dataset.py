import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder

IMAGE_EMB_DIM = 1024  # Swin-base pooler_output


class FusionDataset(Dataset):
    """
    Feature layout per sample:
        [text_embedding]    384-dim   all-MiniLM-L6-v2
        [image_embedding]  1024-dim   Swin-base (zeros if parquet missing)
        [meta + rag]        ~182-dim  MetaEncoder output
    """

    def __init__(
        self,
        split: str,
        encoder: MetaEncoder,
        meta_dir: str = "data/fussion",
        text_emb_dir: str = "src/fussion_branch/embedding/text",
        rag_dir: str = "src/fussion_branch/RAG/return",
        image_emb_dir: Optional[str] = "src/fussion_branch/embedding/image",
        target_col: str = "popularity",
        log_transform_target: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        meta_df = pd.read_csv(f"{meta_dir}/fusion_meta_clean_{split}.csv")
        rag_df  = pd.read_parquet(f"{rag_dir}/rag_features_{split}.parquet")
        text_df = pd.read_parquet(f"{text_emb_dir}/text_embeddings_{split}.parquet")

        meta_df = meta_df.set_index("id")
        rag_df  = rag_df.set_index("id")
        text_df = text_df.set_index("id")

        common_ids = meta_df.index.intersection(rag_df.index).intersection(text_df.index)

        # image embedding (optional — zeros if not available)
        self.use_image = False
        image_df = None
        image_emb_path = (
            f"{image_emb_dir}/image_embeddings_{split}.parquet"
            if image_emb_dir else None
        )
        if image_emb_path and os.path.exists(image_emb_path):
            raw = pd.read_parquet(image_emb_path).set_index("id")
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

        # text (384)
        emb_cols = [c for c in text_df.columns if c.startswith("emb_")]
        self.text_emb = text_df[emb_cols].values.astype(np.float32)

        # image (1024) — zeros if missing
        if self.use_image:
            img_cols = [c for c in image_df.columns if c != "id"]
            self.image_emb = image_df[img_cols].values.astype(np.float32)
        else:
            self.image_emb = np.zeros((N, IMAGE_EMB_DIM), dtype=np.float32)

        # metadata + rag
        self.meta_feat = encoder.transform(meta_df, rag_df)

        # target
        raw_target = meta_df[target_col].values.astype(np.float32)
        if log_transform_target:
            raw_target = np.log1p(raw_target)
        self.target = (raw_target - target_mean) / target_std

    # ── dim properties (used by model constructor) ────────────────────────────
    @property
    def text_dim(self) -> int:
        return self.text_emb.shape[1]

    @property
    def image_dim(self) -> int:
        return self.image_emb.shape[1]

    @property
    def meta_dim(self) -> int:
        return self.meta_feat.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.text_dim + self.image_dim + self.meta_dim

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        features = np.concatenate([
            self.text_emb[idx],
            self.image_emb[idx],
            self.meta_feat[idx],
        ])
        return {
            "features": torch.from_numpy(features),
            "target":   torch.tensor(self.target[idx], dtype=torch.float32),
            "id":       self.ids[idx],
        }
