"""
FusionDatasetE2E: like FusionDataset but returns tokenized descriptions
instead of pre-computed text embeddings — for end-to-end fine-tuning.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder

IMAGE_EMB_DIM = 1024  # Swin-base pooler_output


class FusionDatasetE2E(Dataset):
    """
    Feature layout per sample:
        input_ids        [max_length]   tokenized description
        attention_mask   [max_length]   padding mask
        image_emb        [1024]         Swin-base (zeros if missing)
        meta_feat        [meta_dim]     MetaEncoder output
        target           scalar
    """

    def __init__(
        self,
        split: str,
        encoder: MetaEncoder,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        meta_dir: str = "data/fussion",
        rag_dir: str = "src/fussion_branch/RAG/return",
        image_emb_dir: Optional[str] = "src/fussion_branch/embedding/image",
        target_col: str = "popularity",
        log_transform_target: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        winsor_cap: Optional[float] = None,
    ):
        meta_df = pd.read_csv(f"{meta_dir}/fusion_meta_clean_{split}.csv")
        rag_df  = pd.read_parquet(f"{rag_dir}/rag_features_{split}.parquet")

        meta_df = meta_df.set_index("id")
        rag_df  = rag_df.set_index("id")

        common_ids = meta_df.index.intersection(rag_df.index)

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
            orig_count = len(common_ids)
            if len(img_ids) > 0:
                common_ids = img_ids
                self.use_image = True
                print(f"  [{split}] image embeddings loaded: {len(img_ids)}/{orig_count} rows")
        if not self.use_image:
            print(f"  [{split}] image embeddings not found — using zeros (dim={IMAGE_EMB_DIM})")

        meta_df  = meta_df.loc[common_ids].reset_index()
        rag_df   = rag_df.loc[meta_df["id"]].reset_index()
        if self.use_image:
            image_df = raw.loc[meta_df["id"]].reset_index()

        self.ids = meta_df["id"].values
        N = len(self.ids)

        # tokenize descriptions (null → empty string → all-padding → model learns nothing)
        texts = meta_df["description"].fillna("").tolist()
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids      = encoded["input_ids"]        # (N, max_length)
        self.attention_mask = encoded["attention_mask"]   # (N, max_length)

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
        if winsor_cap is not None:
            raw_target = np.clip(raw_target, None, winsor_cap)
        self.target = (raw_target - target_mean) / target_std

    @property
    def image_dim(self) -> int:
        return self.image_emb.shape[1]

    @property
    def meta_dim(self) -> int:
        return self.meta_feat.shape[1]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "image_emb":      torch.from_numpy(self.image_emb[idx]),
            "meta_feat":      torch.from_numpy(self.meta_feat[idx]),
            "target":         torch.tensor(self.target[idx], dtype=torch.float32),
            "id":             self.ids[idx],
        }
