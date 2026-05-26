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
IMAGE_DIM = 1024  # Swin-base pooler_output（cover 與 char 各自 1024）


def _build_emb_lookup(parquet_path: str, col_prefix: str) -> dict:
    """Load parquet → {id(int): np.ndarray}. Returns {} if file missing."""
    p = Path(parquet_path)
    if not p.exists():
        return {}
    df   = pd.read_parquet(p)
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    return dict(zip(df["id"].astype(int), df[cols].values.astype(np.float32)))


def _load_image_df(path: Optional[str], common_ids) -> Optional[pd.DataFrame]:
    """Load image parquet and intersect with common_ids. Returns None if missing."""
    if not path or not os.path.exists(path):
        return None
    raw = pd.read_parquet(path).set_index("id")
    ids = common_ids.intersection(raw.index)
    return raw.loc[ids] if len(ids) > 0 else None


class FusionDataset(Dataset):
    """
    Returns per-sample dict with separate modality tensors:
        text_emb   (384,)       query anime text embedding
        cover_emb  (1024,)      query anime cover image embedding (zeros if missing)
        char_emb   (1024,)      query anime character embedding via YOLO (zeros if missing)
        meta_feat  (meta_dim,)  MetaEncoder output
        ret_text   (K, 384)     retrieved anime text embeddings  (padded)
        ret_char   (K, 1024)    retrieved anime char embeddings  (padded, for ImageGNN)
        ret_mask   (K,) bool    True = valid retrieved node
        target     scalar
        id         int

    FusionMLP image input = concat([GNN-enhanced char_emb, cover_emb]) → 2048-dim
    ImageGNN operates on char_emb / ret_char (1024-dim).
    """

    def __init__(
        self,
        split: str,
        encoder: MetaEncoder,
        meta_dir: str = "data/fussion",
        meta_suffix: str = "",
        text_emb_dir: str = "src/fussion_branch/embedding/text",
        rag_dir: str = "src/fussion_branch/RAG/return",
        image_emb_dir: Optional[str] = "src/fussion_branch/embedding/image",
        char_emb_dir: Optional[str] = None,
        target_col: str = "popularity",
        log_transform_target: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        winsor_cap: float | None = None,
        top_k_ids: int = 5,
    ):
        self.top_k_ids = top_k_ids

        # ── load primary dataframes ───────────────────────────────────────────
        meta_df = pd.read_csv(f"{meta_dir}/fusion_meta_clean_{split}{meta_suffix}.csv")
        rag_df  = pd.read_parquet(f"{rag_dir}/rag_features_{split}.parquet")
        text_df = pd.read_parquet(f"{text_emb_dir}/text_embeddings_{split}.parquet")

        meta_df = meta_df.set_index("id")
        rag_df  = rag_df.set_index("id")
        text_df = text_df.set_index("id")

        common_ids = meta_df.index.intersection(rag_df.index).intersection(text_df.index)

        # ── cover image embeddings ────────────────────────────────────────────
        cover_path = f"{image_emb_dir}/image_embeddings_{split}.parquet" if image_emb_dir else None
        cover_raw  = _load_image_df(cover_path, common_ids)
        self.use_cover = cover_raw is not None
        if self.use_cover:
            common_ids = common_ids.intersection(cover_raw.index)
            print(f"  [{split}] cover embeddings: {len(common_ids)} rows")
        else:
            print(f"  [{split}] cover embeddings not found — zeros (dim={IMAGE_DIM})")

        # ── character embeddings (YOLO) ───────────────────────────────────────
        char_path = f"{char_emb_dir}/image_embeddings_char_{split}.parquet" if char_emb_dir else None
        char_raw  = _load_image_df(char_path, common_ids)
        self.use_char = char_raw is not None
        if self.use_char:
            common_ids = common_ids.intersection(char_raw.index)
            print(f"  [{split}] char embeddings (YOLO): {len(common_ids)} rows")
        else:
            print(f"  [{split}] char embeddings not found — zeros (dim={IMAGE_DIM})")

        # ── align all frames to common_ids ────────────────────────────────────
        meta_df = meta_df.loc[common_ids].reset_index()
        rag_df  = rag_df.loc[meta_df["id"]].reset_index()
        text_df = text_df.loc[meta_df["id"]].reset_index()

        assert (meta_df["id"].values == rag_df["id"].values).all()
        assert (meta_df["id"].values == text_df["id"].values).all()

        self.ids = meta_df["id"].values
        N = len(self.ids)

        # ── query embeddings ──────────────────────────────────────────────────
        emb_cols = [c for c in text_df.columns if c.startswith("emb_")]
        self.text_emb = text_df[emb_cols].values.astype(np.float32)   # (N, 384)

        if self.use_cover:
            img_cols = [c for c in cover_raw.columns if c != "id"]
            self.cover_emb = cover_raw.loc[meta_df["id"]].reset_index()[img_cols].values.astype(np.float32)
        else:
            self.cover_emb = np.zeros((N, IMAGE_DIM), dtype=np.float32)

        if self.use_char:
            char_cols = [c for c in char_raw.columns if c != "id"]
            self.char_emb = char_raw.loc[meta_df["id"]].reset_index()[char_cols].values.astype(np.float32)
        else:
            self.char_emb = np.zeros((N, IMAGE_DIM), dtype=np.float32)

        # ── metadata + rag features ───────────────────────────────────────────
        self.meta_feat = encoder.transform(meta_df, rag_df)            # (N, meta_dim)

        # ── retrieved_ids for GNN ─────────────────────────────────────────────
        # retrieved anime are always from training set (RAG indexes train only)
        train_text_path = f"{text_emb_dir}/text_embeddings_train.parquet"
        # ImageGNN uses char embeddings; fall back to cover if char not available
        if char_emb_dir:
            train_char_path = f"{char_emb_dir}/image_embeddings_char_train.parquet"
        elif image_emb_dir:
            train_char_path = f"{image_emb_dir}/image_embeddings_train.parquet"
        else:
            train_char_path = ""

        self._text_lookup = _build_emb_lookup(train_text_path, "emb_")
        self._char_lookup = _build_emb_lookup(train_char_path, "img_") if train_char_path else {}

        if "retrieved_ids" in rag_df.columns:
            self._retrieved_ids = [
                json.loads(v) if isinstance(v, str) else []
                for v in rag_df["retrieved_ids"].fillna("[]")
            ]
            print(f"  [{split}] retrieved_ids loaded  "
                  f"(text_lookup={len(self._text_lookup)}, "
                  f"char_lookup={len(self._char_lookup)})")
        else:
            self._retrieved_ids = [[] for _ in range(N)]
            print(f"  [{split}] retrieved_ids not found — GNN will use zero context")

        # ── year lookup for GNN temporal decay ───────────────────────────────
        # Retrieved anime are always from the training set; load train years once.
        train_meta_path = Path(meta_dir) / f"fusion_meta_clean_train{meta_suffix}.csv"
        if train_meta_path.exists():
            _tm = pd.read_csv(train_meta_path, usecols=["id", "release_year"])
            self._year_lookup: dict = dict(zip(
                _tm["id"].astype(int), _tm["release_year"].fillna(0).astype(float)
            ))
        else:
            self._year_lookup = {}
        self._query_years = meta_df["release_year"].fillna(0).values.astype(np.float32)

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
        # When char is available: concat([char, cover]) = 2048
        # When char is disabled (zeros): cover only = 1024
        if self.use_char:
            return self.char_emb.shape[1] + self.cover_emb.shape[1]
        return self.cover_emb.shape[1]

    @property
    def meta_dim(self) -> int:
        return self.meta_feat.shape[1]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        K = self.top_k_ids
        ret_ids = self._retrieved_ids[idx]

        ret_text      = np.zeros((K, TEXT_DIM),  dtype=np.float32)
        ret_char      = np.zeros((K, IMAGE_DIM), dtype=np.float32)
        ret_mask      = np.zeros(K,              dtype=bool)
        ret_year_gaps = np.zeros(K,              dtype=np.float32)

        query_year = float(self._query_years[idx])
        for i, rid in enumerate(ret_ids[:K]):
            if rid in self._text_lookup:
                ret_text[i] = self._text_lookup[rid]
                ret_mask[i] = True
            if rid in self._char_lookup:
                ret_char[i] = self._char_lookup[rid]
            neighbor_year = self._year_lookup.get(int(rid), query_year)
            ret_year_gaps[i] = max(0.0, query_year - neighbor_year)

        return {
            "text_emb":      torch.from_numpy(self.text_emb[idx]),    # (384,)
            "cover_emb":     torch.from_numpy(self.cover_emb[idx]),   # (1024,)
            "char_emb":      torch.from_numpy(self.char_emb[idx]),    # (1024,)
            "meta_feat":     torch.from_numpy(self.meta_feat[idx]),   # (meta_dim,)
            "ret_text":      torch.from_numpy(ret_text),              # (K, 384)
            "ret_char":      torch.from_numpy(ret_char),              # (K, 1024)
            "ret_mask":      torch.from_numpy(ret_mask),              # (K,)
            "ret_year_gaps": torch.from_numpy(ret_year_gaps),         # (K,)
            "target":        torch.tensor(self.target[idx], dtype=torch.float32),
            "id":            int(self.ids[idx]),
        }
