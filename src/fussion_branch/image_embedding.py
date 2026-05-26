"""
Cover and Character image embedders using fine-tuned Swin-base.

  CoverEmbedder      – encodes full anime cover images
  CharacterEmbedder  – encodes pre-cropped character regions (from run_yolo_crop.py)

Both share the same Swin backbone.  dim = 1024 (swin-base-patch4-window7-224).

Usage:
    cover = CoverEmbedder()
    embs  = cover.encode_paths(paths)                   # (N, 1024)

    char  = CharacterEmbedder()
    ids, embs = char.encode_manifest(manifest_df)       # (N_anime, 1024)

Prerequisites for run_yolo_crop.py (YOLO step):
    pip install dghs-imgutils==0.19.0 --no-deps
    CUDA 12 libs visible (conda activate.d/ort_cuda12_libs.sh handles this)
"""
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SwinModel

from src.fussion_branch.image_components.image_process import get_transform_original, ResizeWithPad


def _progress(iterable: Iterable, desc: str = "", total: int = 0) -> Iterator:
    """tqdm in TTY; simple percentage lines otherwise."""
    items = list(iterable)
    n = total or len(items)
    if sys.stderr.isatty():
        yield from tqdm(items, desc=desc, total=n)
    else:
        step = max(1, n // 10)
        for i, item in enumerate(items):
            if i % step == 0:
                print(f"  {desc} {i}/{n} ({100*i//n}%)", flush=True)
            yield item
        print(f"  {desc} {n}/{n} (100%)", flush=True)

_PRETRAINED = "microsoft/swin-base-patch4-window7-224"
_IMAGE_SIZE = 224


class _SwinBase:
    """Shared Swin model loading and encoding logic."""

    def __init__(
        self,
        checkpoint_dir: Optional[str] = "src/fussion_branch/model/best",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_source = (
            checkpoint_dir
            if (checkpoint_dir and Path(checkpoint_dir).exists())
            else _PRETRAINED
        )
        print(f"{self.__class__.__name__}: loading from '{model_source}' on {self.device}")
        self.model = SwinModel.from_pretrained(model_source).to(self.device)
        self.model.eval()

        self.resize    = ResizeWithPad(_IMAGE_SIZE)
        self.transform = get_transform_original(_IMAGE_SIZE)
        self.dim       = self.model.config.hidden_size  # swin-base → 1024

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.transform(self.resize(img))

    @torch.no_grad()
    def _encode_tensor_batch(self, tensors: List[torch.Tensor]) -> np.ndarray:
        batch = torch.stack(tensors).to(self.device)
        embs  = self.model(pixel_values=batch).pooler_output  # (B, 1024)
        return embs.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_paths(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        """List of image paths → (N, 1024) float32 (batched)."""
        all_embs = []
        for i in _progress(range(0, len(paths), batch_size), desc=self.__class__.__name__):
            tensors = []
            for p in paths[i : i + batch_size]:
                fp = Path(p)
                if fp.exists():
                    try:
                        tensors.append(self._preprocess(Image.open(fp).convert("RGB")))
                        continue
                    except Exception:
                        pass
                tensors.append(torch.zeros(3, _IMAGE_SIZE, _IMAGE_SIZE))
            all_embs.append(self._encode_tensor_batch(tensors))
        return np.concatenate(all_embs, axis=0)


class CoverEmbedder(_SwinBase):
    """Encode full anime cover images.

    embs = CoverEmbedder().encode_paths(paths)   # (N, 1024)
    """


class CharacterEmbedder(_SwinBase):
    """Encode pre-cropped character regions and mean-pool per anime.

    Crops are produced by run_yolo_crop.py and indexed via a manifest parquet.

    ids, embs = CharacterEmbedder().encode_manifest(manifest_df)   # (N_anime, 1024)
    crop_embs = CharacterEmbedder().encode_paths(crop_paths)       # (C, 1024), no grouping
    """

    def encode_manifest(
        self,
        manifest: "pd.DataFrame",
        batch_size: int = 64,
    ) -> Tuple[List[int], np.ndarray]:
        """manifest[id, crop_path] → (id_list, embeddings (N_anime, 1024)).

        Encodes all crops in one batched pass, then mean-pools per anime id.
        """
        import pandas as pd  # local import — only needed here

        crop_paths = manifest["crop_path"].tolist()
        crop_ids   = manifest["id"].tolist()

        crop_embs = self.encode_paths(crop_paths, batch_size)  # (C, 1024)

        id_order = list(dict.fromkeys(crop_ids))
        idx_map: dict = {aid: [] for aid in id_order}
        for i, aid in enumerate(crop_ids):
            idx_map[aid].append(i)

        rows = [crop_embs[idx_map[aid]].mean(axis=0)
                for aid in _progress(id_order, desc="mean-pool")]

        return id_order, np.stack(rows, axis=0)
