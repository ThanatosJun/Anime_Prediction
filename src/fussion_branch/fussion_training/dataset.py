import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.fussion_branch.fussion_training.meta_encoder import MetaEncoder

TEXT_DIM  = 384
IMAGE_DIM = 1024  # Swin-base pooler_output


def _build_emb_lookup(parquet_path: str, col_prefix: str) -> dict:
    """Load parquet → {id(int): np.ndarray}. Returns {} if file missing."""
    p = Path(parquet_path)
    if not p.exists():
        return {}
    df   = pd.read_parquet(p)
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    return dict(zip(df["id"].astype(int), df[cols].values.astype(np.float32)))


class FusionDataset(Dataset):
    """
    Returns per-sample dict with separate modality tensors:
        text_emb   (384,)       query anime text embedding
        image_emb  (1024,)      query anime image embedding (zeros if missing)
        meta_feat  (meta_dim,)  MetaEncoder output
        ret_text   (K, 384)     retrieved anime text embeddings  (padded)
        ret_image  (K, 1024)    retrieved anime image embeddings (padded)
        ret_mask   (K,) bool    True = valid retrieved node
        target     scalar
        id         int
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
        winsor_cap: float | None = None,
        top_k_ids: int = 5,
    ):
        self.top_k_ids = top_k_ids

        # ── load primary dataframes ───────────────────────────────────────────
        meta_df = pd.read_csv(f"{meta_dir}/fusion_meta_clean_{split}.csv")
        rag_df  = pd.read_parquet(f"{rag_dir}/rag_features_{split}.parquet")
        text_df = pd.read_parquet(f"{text_emb_dir}/text_embeddings_{split}.parquet")

        meta_df = meta_df.set_index("id")
        rag_df  = rag_df.set_index("id")
        text_df = text_df.set_index("id")

        common_ids = meta_df.index.intersection(rag_df.index).intersection(text_df.index)

        # ── image embeddings (query split) ────────────────────────────────────
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
                self.use_image = True
                print(f"  [{split}] image embeddings: {len(img_ids)} rows")
            else:
                print(f"  [{split}] image embeddings not found — zeros (dim={IMAGE_DIM})")
        if not self.use_image:
            print(f"  [{split}] image embeddings not found — zeros (dim={IMAGE_DIM})")

        # ── align all frames to common_ids ────────────────────────────────────
        meta_df  = meta_df.loc[common_ids].reset_index()
        rag_df   = rag_df.loc[meta_df["id"]].reset_index()
        text_df  = text_df.loc[meta_df["id"]].reset_index()
        if self.use_image:
            image_df = raw.loc[meta_df["id"]].reset_index()

        assert (meta_df["id"].values == rag_df["id"].values).all()
        assert (meta_df["id"].values == text_df["id"].values).all()

        self.ids = meta_df["id"].values
        N = len(self.ids)

        # ── query embeddings ──────────────────────────────────────────────────
        emb_cols = [c for c in text_df.columns if c.startswith("emb_")]
        self.text_emb = text_df[emb_cols].values.astype(np.float32)   # (N, 384)

        if self.use_image:
            img_cols = [c for c in image_df.columns if c != "id"]
            self.image_emb = image_df[img_cols].values.astype(np.float32)  # (N, 1024)
        else:
            self.image_emb = np.zeros((N, IMAGE_DIM), dtype=np.float32)

        # ── metadata + rag features ───────────────────────────────────────────
        self.meta_feat = encoder.transform(meta_df, rag_df)            # (N, meta_dim)

        # ── retrieved_ids for GNN ─────────────────────────────────────────────
        # retrieved anime are always from training set (RAG indexes train only)
        train_text_path  = f"{text_emb_dir}/text_embeddings_train.parquet"
        train_image_path = (
            f"{image_emb_dir}/image_embeddings_train.parquet"
            if image_emb_dir else ""
        )
        self._text_lookup  = _build_emb_lookup(train_text_path,  "emb_")
        self._image_lookup = _build_emb_lookup(train_image_path, "img_") if train_image_path else {}

        if "retrieved_ids" in rag_df.columns:
            self._retrieved_ids = [
                json.loads(v) if isinstance(v, str) else []
                for v in rag_df["retrieved_ids"].fillna("[]")
            ]
            print(f"  [{split}] retrieved_ids loaded  "
                  f"(text_lookup={len(self._text_lookup)}, "
                  f"image_lookup={len(self._image_lookup)})")
        else:
            self._retrieved_ids = [[] for _ in range(N)]
            print(f"  [{split}] retrieved_ids not found — GNN will use zero context")

        # ── target ───────────────────────────────────────────────────────────
        raw_target = meta_df[target_col].values.astype(np.float32)
        if log_transform_target:
            raw_target = np.log1p(raw_target)
        if winsor_cap is not None:
            raw_target = np.clip(raw_target, None, winsor_cap)
        self.target = (raw_target - target_mean) / target_std

    # ── dim properties ────────────────────────────────────────────────────────
    @property
    def text_dim(self) -> int:
        return self.text_emb.shape[1]

    @property
    def image_dim(self) -> int:
        return self.image_emb.shape[1]

    @property
    def meta_dim(self) -> int:
        return self.meta_feat.shape[1]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        K = self.top_k_ids
        ret_ids = self._retrieved_ids[idx]

        # Build retrieved embedding tensors (padded to K)
        ret_text  = np.zeros((K, TEXT_DIM),  dtype=np.float32)
        ret_image = np.zeros((K, IMAGE_DIM), dtype=np.float32)
        ret_mask  = np.zeros(K,              dtype=bool)

        for i, rid in enumerate(ret_ids[:K]):
            if rid in self._text_lookup:
                ret_text[i] = self._text_lookup[rid]
                ret_mask[i] = True
            if rid in self._image_lookup:
                ret_image[i] = self._image_lookup[rid]

        return {
            "text_emb":  torch.from_numpy(self.text_emb[idx]),     # (384,)
            "image_emb": torch.from_numpy(self.image_emb[idx]),    # (1024,)
            "meta_feat": torch.from_numpy(self.meta_feat[idx]),    # (meta_dim,)
            "ret_text":  torch.from_numpy(ret_text),               # (K, 384)
            "ret_image": torch.from_numpy(ret_image),              # (K, 1024)
            "ret_mask":  torch.from_numpy(ret_mask),               # (K,)
            "target":    torch.tensor(self.target[idx], dtype=torch.float32),
            "id":        int(self.ids[idx]),
        }
