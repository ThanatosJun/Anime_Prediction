"""
Sentence-Transformer embedding 產生器（推論用）

來源：src/text_branch/embedding_generator.py（Text branch）
"""

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = 'intfloat/e5-base-v2',
        device: str = 'auto',
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded on {self.device}. Embedding dim: {self.embedding_dim}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        device = (device or 'auto').lower()
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            return 'cpu'
        return device

    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
