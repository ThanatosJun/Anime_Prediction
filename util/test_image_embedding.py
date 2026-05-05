"""
驗證 image model 的輸出維度。

Usage:
    conda activate animeprediction
    python test/test_image_embedding.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from pathlib import Path

from src.image_branch.config import load_config
from src.image_branch.model import get_embedding
from src.image_branch.image_process import load_image, ResizeWithPad, get_transform_original

CHECKPOINT = "results/01/best"
EXPECTED_DIM = 1024


def find_test_images(image_dir: str, n: int = 3):
    paths = []
    for f in Path(image_dir).iterdir():
        if f.suffix == ".jpg":
            paths.append(str(f))
        if len(paths) >= n:
            break
    return paths


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {CHECKPOINT}")

    # 載入微調過的 checkpoint（results/01/best/model.safetensors）
    from transformers import SwinModel
    model = SwinModel.from_pretrained(CHECKPOINT).to(device)
    model.eval()
    print(f"model loaded from fine-tuned checkpoint: {CHECKPOINT}")

    # 取樣幾張圖片
    image_dir = config["data"]["image_dir"]
    image_paths = find_test_images(image_dir, n=3)
    assert image_paths, f"找不到任何圖片於 {image_dir}"

    resize    = ResizeWithPad(config["data"]["image_size"])
    transform = get_transform_original(config["data"]["image_size"])

    # 單張測試
    print("\n=== 單張推論 ===")
    for path in image_paths:
        img = load_image(path)
        assert img is not None, f"無法讀取 {path}"
        tensor = transform(resize(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = get_embedding(model, tensor)
        dim = emb.shape[-1]
        status = "OK" if dim == EXPECTED_DIM else f"FAIL (expected {EXPECTED_DIM})"
        print(f"  {Path(path).name:45s}  shape={tuple(emb.shape)}  [{status}]")

    # Batch 測試
    print("\n=== Batch 推論 ===")
    tensors = []
    for path in image_paths:
        img = load_image(path)
        tensors.append(transform(resize(img)))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        emb_batch = get_embedding(model, batch)
    print(f"  batch shape: {tuple(emb_batch.shape)}  (expected [{len(image_paths)}, {EXPECTED_DIM}])")
    assert emb_batch.shape == (len(image_paths), EXPECTED_DIM), \
        f"FAIL: got {tuple(emb_batch.shape)}"

    # 數值健康檢查
    arr = emb_batch.cpu().numpy()
    print(f"\n=== 數值檢查 ===")
    print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"  NaN: {np.isnan(arr).sum()}  Inf: {np.isinf(arr).sum()}")
    assert not np.isnan(arr).any(), "FAIL: embedding 含有 NaN"
    assert not np.isinf(arr).any(), "FAIL: embedding 含有 Inf"

    print(f"\n所有測試通過，輸出維度 = {EXPECTED_DIM}")


if __name__ == "__main__":
    main()
