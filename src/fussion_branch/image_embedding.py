"""
ImageEmbedder: loads fine-tuned Swin-base and extracts 1024-dim embeddings.

Usage:
    embedder = ImageEmbedder()                         # pretrained only
    embedder = ImageEmbedder(checkpoint_dir="results/01/best")  # fine-tuned

    emb = embedder.encode_path("data/image/12345_coverImage_medium.jpg")  # (1024,)
    embs = embedder.encode_paths([...])                                    # (N, 1024)
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import SwinModel

from src.fussion_branch.image_components.image_process import get_transform_original, ResizeWithPad

_PRETRAINED = "microsoft/swin-base-patch4-window7-224"
_IMAGE_SIZE = 224


class ImageEmbedder:
    def __init__(
        self,
        checkpoint_dir: Optional[str] = "results/01/best",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_source = checkpoint_dir if (checkpoint_dir and Path(checkpoint_dir).exists()) else _PRETRAINED
        print(f"ImageEmbedder: loading from '{model_source}' on {self.device}")
        self.model = SwinModel.from_pretrained(model_source).to(self.device)
        self.model.eval()

        self.resize    = ResizeWithPad(_IMAGE_SIZE)
        self.transform = get_transform_original(_IMAGE_SIZE)
        # detect actual output dim from the loaded model config
        self.dim = self.model.config.hidden_size  # swin-base = 1024

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        """PIL Image → (1024,) float32."""
        img    = self.resize(img)
        tensor = self.transform(img).unsqueeze(0).to(self.device)   # (1, 3, 224, 224)
        emb    = self.model(pixel_values=tensor).pooler_output       # (1, 1024)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_path(self, path: str) -> np.ndarray:
        """Image file path → (1024,) float32.  Returns zeros if file missing."""
        p = Path(path)
        if not p.exists():
            return np.zeros(self.dim, dtype=np.float32)
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            return np.zeros(self.dim, dtype=np.float32)
        return self.encode_image(img)

    @torch.no_grad()
    def encode_paths(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        """List of image paths → (N, 1024) float32."""
        all_embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            tensors = []
            for p in batch_paths:
                fp = Path(p)
                if fp.exists():
                    try:
                        img = Image.open(fp).convert("RGB")
                        img = self.resize(img)
                        tensors.append(self.transform(img))
                        continue
                    except Exception:
                        pass
                tensors.append(torch.zeros(3, _IMAGE_SIZE, _IMAGE_SIZE))
            batch = torch.stack(tensors).to(self.device)
            embs  = self.model(pixel_values=batch).pooler_output  # (B, 1024)
            all_embs.append(embs.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)
