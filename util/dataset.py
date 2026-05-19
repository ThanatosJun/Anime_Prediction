import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from util.image_process import load_image, ResizeWithPad
from src.YOLO import detect_person, detect_faces
class AnimeImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        image_col: str,
        transform_orig,
        transform_aug,
        use_yolo=False
    ):
        self.image_dir     = image_dir
        self.image_col     = image_col
        self.transform_orig = transform_orig
        self.transform_aug  = transform_aug
        self.resize        = ResizeWithPad(224)
        self.use_yolo      = use_yolo
        if use_yolo:
            from src.config import load_yolo_config
            self._yolo_cfg = load_yolo_config()

        # 只保留圖片實際存在的 row，避免 dummy tensor 污染訓練
        df = df.reset_index(drop=True)
        mask = df['id'].apply(
            lambda idx: os.path.isfile(os.path.join(image_dir, f"{idx}_{image_col}.jpg"))
        )
        self.df = df[mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row  = self.df.iloc[i]
        idx  = int(row['id'])
        path = os.path.join(self.image_dir, f"{idx}_{self.image_col}.jpg")

        img = load_image(path)
        if img is None:
            dummy = torch.zeros(3, 224, 224)
            return dummy, dummy, idx

        if self.use_yolo:
            cfg = self._yolo_cfg
            detect_mode = cfg.get('detect_mode', 'person')
            det_cfg = cfg['face_detection'] if detect_mode == 'face' else cfg['detection']
            max_det = det_cfg['max_detections']

            # upscale 小圖以提升偵測率
            w, h = img.size
            scale = max(640 / w, 640 / h)
            if scale > 1:
                img_det = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            else:
                img_det = img

            results = []
            if detect_mode in ('person', 'both'):
                m, d = cfg['model'], cfg['detection']
                results += detect_person(
                    img_det,
                    level=m['level'], version=m['version'],
                    conf_threshold=d['conf_threshold'], iou_threshold=d['iou_threshold'],
                )
            if detect_mode in ('face', 'both'):
                m, d = cfg['face_model'], cfg['face_detection']
                results += detect_faces(
                    img_det,
                    level=m['level'], version=m['version'],
                    conf_threshold=d['conf_threshold'], iou_threshold=d['iou_threshold'],
                )
            results = sorted(results, key=lambda x: x[2], reverse=True)[:max_det]
            frame_ = []  # 儲存所有 crop
            if results:
                for j, (bbox, label, conf) in enumerate(results):
                    x0, y0, x1, y1 = bbox
                    crop = img_det.crop((x0, y0, x1, y1))
                    frame_.append(crop)
                    if det_cfg['save_crops']:
                        save_dir = cfg['data']['output_dir']
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{idx}_{self.image_col}_crop_{j}.jpg")
                        crop.save(save_path)
            else:
                frame_ = [img_det]  # fallback：無偵測結果時用整張圖

            # 對每個 crop 做 resize + transform，最後 stack
            origs, augs = [], []
            for crop in frame_:
                c = self.resize(crop)
                origs.append(self.transform_orig(c))
                augs.append(self.transform_aug(c))
            orig = torch.stack(origs)  # (N, 3, 224, 224)
            aug  = torch.stack(augs)   # (N, 3, 224, 224)
            return orig, aug, idx

        # use_yolo=False 時：單張圖，維持 (3, 224, 224)
        img  = self.resize(img)
        orig = self.transform_orig(img)
        aug  = self.transform_aug(img)
        return orig, aug, idx


def yolo_collate_fn(batch):
    """(orig_A, aug_A, idx_A)"""
    origs = [item[0] for item in batch]  # List of (N_i, 3, 224, 224)
    augs  = [item[1] for item in batch]  # List of (N_i, 3, 224, 224)
    idxs  = [item[2] for item in batch]
    return origs, augs, idxs


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, use_yolo: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=yolo_collate_fn if use_yolo else None,
    )
