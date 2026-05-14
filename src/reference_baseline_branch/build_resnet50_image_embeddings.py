from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


IMAGE_COLUMNS = ("coverImage_medium", "bannerImage")
FEATURE_DIM = 2048


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ResNet-50 avg-pool image features for C1 reference baselines."
    )
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
        help="Reference baseline YAML config.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split subset, e.g. train val test.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--weights",
        default="imagenet",
        choices=["imagenet", "none"],
        help="Use ImageNet pretrained weights by default, matching the ResNet-50 proxy intent.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split row limit for smoke tests.",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    data_cfg = config["data"]
    splits = args.splits or data_cfg.get("splits", ["train", "val", "test"])
    image_dir = Path(data_cfg.get("image_dir", "data/image"))
    output_dir = Path(data_cfg["resnet50_image_emb_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    model, transform = _load_model(args.weights, device)

    for split in splits:
        meta_path = Path(data_cfg["meta_dir"]) / f"fusion_meta_clean_{split}.csv"
        df = pd.read_csv(meta_path, usecols=[data_cfg.get("id_col", "id")])
        if args.limit is not None:
            df = df.head(args.limit)

        ids = df[data_cfg.get("id_col", "id")].astype(int).tolist()
        paths = _collect_paths(ids, image_dir)
        print(f"[{split}] ids={len(ids)} available_images={len(paths)}")
        embeddings = _encode_paths(paths, model, transform, device, args.batch_size)
        table = _build_split_table(ids, image_dir, embeddings)

        output_path = output_dir / f"resnet50_image_embeddings_{split}.parquet"
        table.to_parquet(output_path, index=False)
        print(f"[{split}] saved {output_path} shape={table.shape}")


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _load_model(weights_name: str, device: torch.device) -> tuple[nn.Module, object]:
    weights = ResNet50_Weights.DEFAULT if weights_name == "imagenet" else None
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    transform = ResNet50_Weights.DEFAULT.transforms()
    return model, transform


def _collect_paths(ids: Sequence[int], image_dir: Path) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for anime_id in ids:
        for column in IMAGE_COLUMNS:
            path = _image_path(image_dir, anime_id, column)
            if path.exists() and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _encode_paths(
    paths: Sequence[Path],
    model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
) -> Dict[Path, np.ndarray]:
    embeddings: Dict[Path, np.ndarray] = {}
    for start in tqdm(range(0, len(paths), batch_size), desc="resnet50"):
        batch_paths = paths[start : start + batch_size]
        tensors: List[torch.Tensor] = []
        valid_paths: List[Path] = []
        for path in batch_paths:
            tensor = _load_tensor(path, transform)
            if tensor is None:
                continue
            tensors.append(tensor)
            valid_paths.append(path)
        if not tensors:
            continue
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            batch_emb = model(batch).detach().cpu().numpy().astype(np.float32)
        for path, emb in zip(valid_paths, batch_emb):
            embeddings[path] = emb
    return embeddings


def _load_tensor(path: Path, transform) -> torch.Tensor | None:
    try:
        with Image.open(path) as img:
            return transform(img.convert("RGB"))
    except (OSError, UnidentifiedImageError):
        return None


def _build_split_table(
    ids: Sequence[int],
    image_dir: Path,
    embeddings: Dict[Path, np.ndarray],
) -> pd.DataFrame:
    feature_names = _feature_names()
    matrix = np.zeros((len(ids), len(feature_names)), dtype=np.float32)

    for row_idx, anime_id in enumerate(ids):
        offset = 0
        availability: List[float] = []
        for column in IMAGE_COLUMNS:
            path = _image_path(image_dir, anime_id, column)
            emb = embeddings.get(path)
            if emb is not None:
                matrix[row_idx, offset : offset + FEATURE_DIM] = emb
                availability.append(1.0)
            else:
                availability.append(0.0)
            offset += FEATURE_DIM
        matrix[row_idx, offset : offset + len(availability)] = np.asarray(availability, dtype=np.float32)

    table = pd.DataFrame(matrix, columns=feature_names)
    table.insert(0, "id", np.asarray(ids, dtype=np.int64))
    return table


def _feature_names() -> List[str]:
    names: List[str] = []
    for label in ("cover", "banner"):
        names.extend(f"resnet_{label}_{idx:04d}" for idx in range(FEATURE_DIM))
    names.extend(["resnet_cover_available", "resnet_banner_available"])
    return names


def _image_path(image_dir: Path, anime_id: int, column: str) -> Path:
    return image_dir / f"{anime_id}_{column}.jpg"


if __name__ == "__main__":
    main()
