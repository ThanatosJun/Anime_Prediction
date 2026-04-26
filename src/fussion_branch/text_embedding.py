"""
TextEmbedder: thin wrapper that fully reuses text_components.

Config:  src/fussion_branch/text_components/embedding_config.yaml
Components:
  TextPreprocessor   (text_components/text_preprocessor.py)
  EmbeddingGenerator (text_components/embedding_generator.py)

Usage:
    embedder = TextEmbedder()

    emb  = embedder.encode("A young hero fights monsters.")  # (384,)
    embs = embedder.encode(["text1", None, "text2"])         # (3, 384); None row = zeros
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import yaml

from src.fussion_branch.text_components.embedding_generator import EmbeddingGenerator
from src.fussion_branch.text_components.text_preprocessor import TextPreprocessor

_CONFIG_PATH = Path("src/fussion_branch/text_components/embedding_config.yaml")


def _load_text_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TextEmbedder:
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        cfg = _load_text_config(Path(config_path) if config_path else _CONFIG_PATH)

        emb_cfg = cfg.get("embedding", {})
        pre_cfg = cfg.get("preprocessing", {})

        self.preprocessor = TextPreprocessor(
            lowercase=pre_cfg.get("lowercase", True),
            remove_urls=pre_cfg.get("remove_urls", True),
            remove_extra_whitespace=pre_cfg.get("remove_extra_whitespace", True),
            min_length=int(pre_cfg.get("min_length", 10)),
            max_length=int(pre_cfg.get("max_length", 512)),
        )
        self.generator = EmbeddingGenerator(
            model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=device or emb_cfg.get("device", "auto"),
            batch_size=int(emb_cfg.get("batch_size", 64)),
            random_seed=int(cfg.get("random_seed", 42)),
        )
        self.dim = self.generator.embedding_dim

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        str  → (384,) float32
        list → (N, 384) float32
        None / too-short / invalid → zeros row (matches text_branch pipeline)
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        cleaned    = [self.preprocessor.clean(t) for t in texts]
        valid_idx  = [i for i, c in enumerate(cleaned) if c is not None]
        valid_text = [cleaned[i] for i in valid_idx]

        result = np.zeros((len(texts), self.dim), dtype=np.float32)

        if valid_text:
            embs = self.generator.encode(valid_text, show_progress_bar=show_progress)
            for out_i, emb in zip(valid_idx, embs):
                result[out_i] = emb.astype(np.float32)

        return result[0] if single else result
