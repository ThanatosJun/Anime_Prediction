import os
import tempfile

import numpy as np
import torch
import requests
from transformers import SwinModel

from util.image_process import load_image, ResizeWithPad, get_transform_original


class ImageEmbedder:
    def __init__(self, model_path: str, config: dict):
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model     = SwinModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.resize    = ResizeWithPad(224)
        self.transform = get_transform_original(config['data']['image_size'])

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

    def embed_batch(self, image_paths: list) -> np.ndarray:
        tensors = []
        for path in image_paths:
            img = load_image(path)
            if img is None:
                tensors.append(torch.zeros(3, 224, 224))
            else:
                img = self.resize(img)
                tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            output = self.model(pixel_values=batch)
        return output.pooler_output.cpu().numpy()

    def embed_url(self, url: str) -> np.ndarray:
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
