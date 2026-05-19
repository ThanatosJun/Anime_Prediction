"""
ImageEmbedder: loads fine-tuned Swin-base and extracts 1024-dim embeddings.

Usage:
    embedder = ImageEmbedder()                                              # pretrained only
    embedder = ImageEmbedder(checkpoint_dir="src/fussion_branch/model/best")  # fine-tuned
    embedder = ImageEmbedder(use_yolo=True)                                 # with YOLO preprocessing

    emb = embedder.encode_path("data/image/12345_coverImage_extraLarge.jpg")  # (1024,)
    embs = embedder.encode_paths([...])                                    # (N, 1024)

When use_yolo=True:
    - Detects anime characters via YOLO and crops each detected region
    - Encodes all crops and mean-pools into a single (1024,) embedding
    - Falls back to the full image when no characters are detected
    - Recommended only for coverImage_extraLarge; set use_yolo=False for bannerImage
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import SwinModel

from src.fussion_branch.image_components.image_process import get_transform_original, ResizeWithPad

_PRETRAINED = "microsoft/swin-base-patch4-window7-224"
_IMAGE_SIZE = 224

# Default YOLO detection params (can be overridden via yolo_cfg dict)
_YOLO_DEFAULTS = {
    "level":               "m",
    "version":             "v1.1",
    "conf_threshold":      0.2,
    "iou_threshold":       0.8,
    "max_persons":         5,
    "fallback_full_image": True,
}


class ImageEmbedder:
    def __init__(
        self,
        checkpoint_dir: Optional[str] = "src/fussion_branch/model/best",
        device: Optional[str] = None,
        use_yolo: bool = False,
        yolo_cfg: Optional[dict] = None,
    ):
        """
        Args:
            checkpoint_dir: Fine-tuned Swin checkpoint dir; falls back to pretrained if missing.
            device:         "cuda" / "cpu". Auto-detected when None.
            use_yolo:       Apply YOLO character detection before encoding.
            yolo_cfg:       Override YOLO detection params (see _YOLO_DEFAULTS).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_source = checkpoint_dir if (checkpoint_dir and Path(checkpoint_dir).exists()) else _PRETRAINED
        print(f"ImageEmbedder: loading from '{model_source}' on {self.device}")
        self.model = SwinModel.from_pretrained(model_source).to(self.device)
        self.model.eval()

        self.resize    = ResizeWithPad(_IMAGE_SIZE)
        self.transform = get_transform_original(_IMAGE_SIZE)
        self.dim = self.model.config.hidden_size  # swin-base = 1024

        self.use_yolo = use_yolo
        if use_yolo:
            from src.fussion_branch.image_components.yolo_detector import detect_and_crop
            self._detect_and_crop = detect_and_crop
            self._yolo_cfg = {**_YOLO_DEFAULTS, **(yolo_cfg or {})}
            print(f"ImageEmbedder: YOLO enabled  "
                  f"(level={self._yolo_cfg['level']}, "
                  f"conf={self._yolo_cfg['conf_threshold']}, "
                  f"max_persons={self._yolo_cfg['max_persons']})")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _encode_tensor_batch(self, tensors: List[torch.Tensor]) -> np.ndarray:
        """Stack a list of (3, 224, 224) tensors and forward through Swin.

        Returns (B, 1024) float32 numpy.
        """
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self.model(pixel_values=batch).pooler_output  # (B, 1024)
        return embs.cpu().numpy().astype(np.float32)

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """PIL Image → (3, 224, 224) tensor (resize + pad + normalise)."""
        return self.transform(self.resize(img))

    def _yolo_crops(self, img: Image.Image) -> List[Image.Image]:
        """Return YOLO-detected character crops (or [img] as fallback)."""
        return self._detect_and_crop(img, **self._yolo_cfg)

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        """PIL Image → (1024,) float32.

        With YOLO: detects characters, encodes all crops, mean-pools.
        Without YOLO: encodes the full image directly.
        """
        if self.use_yolo:
            crops = self._yolo_crops(img)
            tensors = [self._preprocess(c) for c in crops]
            embs = self._encode_tensor_batch(tensors)  # (N_crops, 1024)
            return embs.mean(axis=0)                   # (1024,)

        tensor = self._preprocess(img).unsqueeze(0).to(self.device)
        emb    = self.model(pixel_values=tensor).pooler_output  # (1, 1024)
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

    def encode_paths(self, paths: List[str], batch_size: int = 64) -> np.ndarray:
        """List of image paths → (N, 1024) float32.

        Without YOLO: batched forward for speed.
        With YOLO: processes image-by-image (each may produce multiple crops).
        """
        if self.use_yolo:
            return self._encode_paths_yolo(paths)
        return self._encode_paths_batch(paths, batch_size)

    # ── private batch helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_paths_batch(self, paths: List[str], batch_size: int) -> np.ndarray:
        """Standard batched encoding (no YOLO)."""
        all_embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            tensors = []
            for p in batch_paths:
                fp = Path(p)
                if fp.exists():
                    try:
                        img = Image.open(fp).convert("RGB")
                        tensors.append(self._preprocess(img))
                        continue
                    except Exception:
                        pass
                tensors.append(torch.zeros(3, _IMAGE_SIZE, _IMAGE_SIZE))
            embs = self._encode_tensor_batch(tensors)  # (B, 1024)
            all_embs.append(embs)
        return np.concatenate(all_embs, axis=0)

    def _encode_paths_yolo(self, paths: List[str]) -> np.ndarray:
        """YOLO-aware encoding: detect crops per image, encode, mean-pool."""
        all_embs = []
        for p in paths:
            fp = Path(p)
            if not fp.exists():
                all_embs.append(np.zeros(self.dim, dtype=np.float32))
                continue
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                all_embs.append(np.zeros(self.dim, dtype=np.float32))
                continue
            all_embs.append(self.encode_image(img))  # handles YOLO + mean-pool
        return np.stack(all_embs, axis=0)             # (N, 1024)
