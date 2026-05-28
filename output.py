import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import requests
from PIL import Image
from transformers import SwinModel

from src.config import load_config, load_yolo_config
from src.model import get_stage_embeddings
from src.YOLO import detect_person, detect_faces
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
        self.use_stage = config['model'].get('stage', False)

        yolo_config    = load_yolo_config()
        self.use_yolo  = yolo_config.get('yolo', {}).get('use', False)
        self._yolo_cfg = yolo_config

    def _get_yolo_crops(self, img: Image.Image) -> List[Image.Image]:
        """偵測並回傳裁切後的 PIL Image list；無結果時 fallback 整張圖。"""
        w, h = img.size
        scale = max(640 / w, 640 / h)
        if scale > 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        detect_mode = self._yolo_cfg.get('detect_mode', 'person')
        results = []
        if detect_mode in ('person', 'both'):
            m, d = self._yolo_cfg['model'], self._yolo_cfg['detection']
            results += detect_person(img, level=m['level'], version=m['version'],
                                     conf_threshold=d['conf_threshold'], iou_threshold=d['iou_threshold'])
        if detect_mode in ('face', 'both'):
            m, d = self._yolo_cfg['face_model'], self._yolo_cfg['face_detection']
            results += detect_faces(img, level=m['level'], version=m['version'],
                                    conf_threshold=d['conf_threshold'], iou_threshold=d['iou_threshold'])

        det_key = 'face_detection' if detect_mode == 'face' else 'detection'
        max_det = self._yolo_cfg[det_key]['max_detections']
        results = sorted(results, key=lambda x: x[2], reverse=True)[:max_det]

        if not results:
            return [img]
        return [img.crop(bbox) for (bbox, _, _) in results]

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        return self.transform(self.resize(img))  # (3, 224, 224)

    def _forward(self, batch: torch.Tensor):
        """batch: (N, 3, 224, 224)
          use_stage=False → np.ndarray (1024,)
          use_stage=True  → [np.ndarray(128,), ndarray(256,), ndarray(512,), ndarray(1024,)]
        """
        with torch.no_grad():
            if self.use_stage:
                embs = get_stage_embeddings(self.model, batch)  # [(N,C), ...]
                return [e.mean(dim=0).cpu().numpy() for e in embs]
            else:
                return self.model(pixel_values=batch).pooler_output.mean(dim=0).cpu().numpy()

    def embed(self, image_path: str):
        img = load_image(image_path)
        if img is None:
            return None
        imgs = self._get_yolo_crops(img) if self.use_yolo else [img]
        batch = torch.stack([self._preprocess(i) for i in imgs]).to(self.device)
        return self._forward(batch)

    def embed_batch(self, image_paths: List[str]) -> List:
        return [self.embed(path) for path in image_paths]

    def embed_url(self, url: str):
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
            if emb is None:
                results[idx] = None
            elif self.use_stage:
                results[idx] = [e.tolist() for e in emb]  # [list(128), list(256), list(512), list(1024)]
            else:
                results[idx] = emb.tolist()
        return results

    def save_embeddings(
        self,
        cover_embs: Dict[int, Optional[list]],
        banner_embs: Dict[int, Optional[list]],
        output_path: str = None,
    ):
        all_idx = sorted(set(cover_embs) | set(banner_embs))
        records = []
        for idx in all_idx:
            cover_val  = cover_embs.get(idx)
            banner_val = banner_embs.get(idx)
            if self.use_stage:
                row = {'idx': idx}
                for s in range(4):
                    row[f'coverImage_emb_s{s}']  = cover_val[s]  if cover_val  is not None else None
                    row[f'bannerImage_emb_s{s}'] = banner_val[s] if banner_val is not None else None
            else:
                row = {
                    'idx':             idx,
                    'coverImage_emb':  cover_val,
                    'bannerImage_emb': banner_val,
                }
            records.append(row)
        df = pd.DataFrame(records)
        path = Path(output_path or self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def main():
    config   = load_config()
    inf_cfg  = config['inference']
    model_path = os.path.join(inf_cfg['results_dir'], inf_cfg['run_id'], 'best')
    embedder = ImageEmbedder(model_path=model_path, config=config)
    if 'use_yolo' in inf_cfg:
        embedder.use_yolo = inf_cfg['use_yolo']

    file_kind = inf_cfg.get('file_kind', 'file')

    if file_kind == 'file':
        df = pd.read_csv(config['data']['csv_path'])
        image_dir = config['data']['image_dir']
        cover_embs  = embedder.embed_dataframe(df, image_dir, 'coverImage_medium')
        banner_embs = embedder.embed_dataframe(df, image_dir, 'bannerImage')
        embedder.save_embeddings(cover_embs, banner_embs, inf_cfg['embedding_path'])
        print(f"Saved → {inf_cfg['embedding_path']}")

    elif file_kind == 'image':
        if len(sys.argv) < 2:
            print("Usage: python output.py <image_path>")
            sys.exit(1)
        image_path = sys.argv[1]
        emb = embedder.embed(image_path)
        if emb is None:
            print(f"Failed to load: {image_path}")
            sys.exit(1)
        if embedder.use_stage:
            for i, e in enumerate(emb):
                print(f"Stage {i}: shape={e.shape}")
        else:
            print(f"Embedding shape: {emb.shape}")
            print(emb)

    else:
        print(f"Unknown file_kind: {file_kind!r}  (expected 'file' or 'image')")
        sys.exit(1)


if __name__ == '__main__':
    main()
