import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from util.image_process import load_image, ResizeWithPad


class AnimeImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        image_col: str,
        transform_orig,
        transform_aug,
    ):
        self.image_dir     = image_dir
        self.image_col     = image_col
        self.transform_orig = transform_orig
        self.transform_aug  = transform_aug
        self.resize        = ResizeWithPad(224)

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

        img  = self.resize(img)
        orig = self.transform_orig(img)
        aug  = self.transform_aug(img)
        return orig, aug, idx


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
