import os
import csv

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

from src.image_branch.config import load_config


def make_image_dir(image_dir: str) -> None:
    Path(image_dir).mkdir(parents=True, exist_ok=True)


def fetch_one(url: str, save_path: str, timeout: int = 10) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception:
        return False


def log_result(log_path: str, idx, col: str, url: str, status: str) -> None:
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([idx, col, url, status])


def filter_by_ratio(df: pd.DataFrame, ratio: float, seed: int = 42) -> pd.DataFrame:
    if ratio >= 1.0:
        return df
    return df.sample(frac=ratio, random_state=seed).reset_index(drop=True)


def getImage(config: dict) -> None:
    data_cfg   = config['data']
    csv_path   = data_cfg['csv_path']
    image_cols = data_cfg['image_columns']
    ratio      = data_cfg['fetch_ratio']
    image_dir  = data_cfg['image_dir']
    log_path   = data_cfg['log_path']

    make_image_dir(image_dir)

    df = pd.read_csv(csv_path)
    df = filter_by_ratio(df, ratio)

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Fetching images'):
        idx = row['id']
        for col in image_cols:
            url = str(row.get(col, '')).strip()
            if not url or not url.startswith('http'):
                log_result(log_path, idx, col, url, 'skip')
                continue
            save_path = os.path.join(image_dir, f"{idx}_{col}.jpg")
            success = fetch_one(url, save_path)
            log_result(log_path, idx, col, url, 'success' if success else 'error')
