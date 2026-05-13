import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import requests
from transformers import SwinModel

from src.config import load_config
from util.image_process import load_image, ResizeWithPad, get_transform_original


class ImageEmbedder:
    def __init__(self, model_path: str, config: dict = None):
        if config is None:
            config = load_config()
        self.device    = torch.device(
            config['training']['device'] if torch.cuda.is_available() else 'cpu'
        )
        self.model     = SwinModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.resize    = ResizeWithPad(config['data']['image_size'])
        self.transform = get_transform_original(config['data']['image_size'])
        self.output_path = config['output']['embedding_path']

    def _preprocess(self, img):
        img    = self.resize(img)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def embed(self, image_path: str) -> np.ndarray:
        img = load_image(image_path)
        if img is None:
            return None
        tensor = self._preprocess(img)
        with torch.no_grad():
            output = self.model(pixel_values=tensor)
        return output.pooler_output.squeeze(0).cpu().numpy()

    def embed_batch(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        valid_indices = []
        tensors = []
        for i, path in enumerate(image_paths):
            img = load_image(path)
            if img is not None:
                img = self.resize(img)
                tensors.append(self.transform(img))
                valid_indices.append(i)

        results: List[Optional[np.ndarray]] = [None] * len(image_paths)
        if tensors:
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                output = self.model(pixel_values=batch)
            embs = output.pooler_output.cpu().numpy()
            for out_i, src_i in enumerate(valid_indices):
                results[src_i] = embs[out_i]
        return results

    def embed_url(self, url: str) -> Optional[np.ndarray]:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(tmp_path, 'wb') as f:
                f.write(response.content)
            return self.embed(tmp_path)
        finally:
            os.remove(tmp_path)

    def embed_dataframe(
        self, df: pd.DataFrame, image_dir: str, col: str
    ) -> Dict[int, Optional[list]]:
        image_dir = Path(image_dir)
        results = {}
        for _, row in df.iterrows():
            idx = row['id']
            emb = self.embed(str(image_dir / f"{idx}_{col}.jpg"))
            results[idx] = emb.tolist() if emb is not None else None
        return results

    def save_embeddings(
        self,
        cover_embs: Dict[int, Optional[list]],
        banner_embs: Dict[int, Optional[list]],
        output_path: str = None,
    ):
        all_idx = sorted(set(cover_embs) | set(banner_embs))
        records = [
            {
                'idx': idx,
                'coverImage_emb': cover_embs.get(idx),
                'bannerImage_emb': banner_embs.get(idx),
            }
            for idx in all_idx
        ]
        df = pd.DataFrame(records)
        path = Path(output_path or self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
