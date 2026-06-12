"""Build MAL 2025 external image embeddings with YOLO crops filled.

The original MAL 2025 external image parquet has the same cover/banner/yolo
schema as the CARMA pipeline, but its yolo slot is zero-filled. This script
keeps the existing cover and banner columns, embeds YOLO crops from the local
MAL cover images, and writes a new image parquet plus copied split metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import SwinModel

ROOT = Path(__file__).resolve().parents[2]
COMPONENT_IMAGE = ROOT / "src_2" / "component_image"
if str(COMPONENT_IMAGE) not in sys.path:
    sys.path.insert(0, str(COMPONENT_IMAGE))

from YOLO import detect_faces, detect_person  # noqa: E402
from image_process import ResizeWithPad, get_transform_original, load_image  # noqa: E402


DEFAULT_SPLITS = ["mal2025_popularity_local_ready", "mal2025_dual_local_ready"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill MAL 2025 yolo image embeddings from cover images.")
    parser.add_argument("--config", default="src_2/component_image/image_encoder_config.yaml")
    parser.add_argument("--model-path", default="src_2/component_image/model-image/best")
    parser.add_argument("--cover-dir", default="data/external_assets/mal2025_image/cover")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--suffix", default="yolo")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per split.")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _model_source(model_path: str | Path) -> str:
    raw = str(model_path)
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return str(path)
    candidate = ROOT / path
    if candidate.exists():
        return str(candidate)
    return raw


def _load_model(model_path: str | Path, config: dict) -> tuple[SwinModel, ResizeWithPad, object, torch.device]:
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    model = SwinModel.from_pretrained(_model_source(model_path)).to(device)
    model.eval()
    resize = ResizeWithPad(config["data"]["image_size"])
    transform = get_transform_original(config["data"]["image_size"])
    return model, resize, transform, device


def _get_crops(img: Image.Image, yolo_cfg: dict) -> list[Image.Image]:
    w, h = img.size
    scale = max(640 / w, 640 / h)
    if scale > 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    detect_mode = yolo_cfg.get("detect_mode", "both")
    results = []
    if detect_mode in ("person", "both"):
        m, d = yolo_cfg["model"], yolo_cfg["detection"]
        results += detect_person(
            img,
            level=m["level"],
            version=m["version"],
            conf_threshold=d["conf_threshold"],
            iou_threshold=d["iou_threshold"],
        )
    if detect_mode in ("face", "both"):
        m, d = yolo_cfg["face_model"], yolo_cfg["face_detection"]
        results += detect_faces(
            img,
            level=m["level"],
            version=m["version"],
            conf_threshold=d["conf_threshold"],
            iou_threshold=d["iou_threshold"],
        )

    det_key = "face_detection" if detect_mode == "face" else "detection"
    max_det = yolo_cfg[det_key]["max_detections"]
    results = sorted(results, key=lambda x: x[2], reverse=True)[:max_det]
    if not results:
        return [img]
    return [img.crop(bbox) for (bbox, _, _) in results]


def _cover_path(cover_dir: Path, anime_id: int) -> Path:
    return cover_dir / f"mal2025_{anime_id}_coverImage_extraLarge.jpg"


def _cover_path_map(split: str) -> dict[int, Path]:
    id_map_path = ROOT / "data" / "external_transformed" / f"{split}_id_map.csv"
    if not id_map_path.exists():
        return {}
    id_map = pd.read_csv(id_map_path)
    if "id" not in id_map.columns or "local_cover_image_path" not in id_map.columns:
        return {}
    paths: dict[int, Path] = {}
    for row in id_map[["id", "local_cover_image_path"]].itertuples(index=False):
        if pd.isna(row.local_cover_image_path):
            continue
        paths[int(row.id)] = Path(str(row.local_cover_image_path))
    return paths


def _iter_batches(items: list[tuple[int, list[Image.Image]]], batch_size: int):
    batch_ids: list[int] = []
    batch_imgs: list[Image.Image] = []
    owner: list[int] = []
    for anime_id, crops in items:
        for crop in crops:
            batch_imgs.append(crop)
            owner.append(anime_id)
        batch_ids.append(anime_id)
        if len(batch_imgs) >= batch_size:
            yield batch_imgs, owner
            batch_imgs, owner = [], []
    if batch_imgs:
        yield batch_imgs, owner


def _embed_items(
    items: list[tuple[int, list[Image.Image]]],
    model: SwinModel,
    resize: ResizeWithPad,
    transform,
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    with torch.no_grad():
        for batch_imgs, owner in _iter_batches(items, batch_size):
            batch = torch.stack([transform(resize(img)) for img in batch_imgs]).to(device)
            embs = model(pixel_values=batch).pooler_output.detach().cpu().numpy()
            for anime_id, emb in zip(owner, embs):
                sums[anime_id] = sums.get(anime_id, np.zeros_like(emb)) + emb
                counts[anime_id] = counts.get(anime_id, 0) + 1
    return {anime_id: sums[anime_id] / counts[anime_id] for anime_id in sums}


def _write_split_copy(split: str, new_split: str) -> None:
    dataset_dir = ROOT / "src_2" / "data" / "dataset"
    src = dataset_dir / f"fusion_meta_clean_{split}_v2.csv"
    dst = dataset_dir / f"fusion_meta_clean_{new_split}_v2.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    dst.write_bytes(src.read_bytes())

    rag_dir = ROOT / "src_2" / "RAG" / "return"
    rag_src = rag_dir / f"rag_features_{split}.parquet"
    rag_dst = rag_dir / f"rag_features_{new_split}.parquet"
    if rag_src.exists():
        rag_dst.write_bytes(rag_src.read_bytes())

    text_dir = ROOT / "src_2" / "embedding" / "text"
    text_src = text_dir / f"text_embeddings_{split}.parquet"
    text_dst = text_dir / f"text_embeddings_{new_split}.parquet"
    if text_src.exists():
        text_dst.write_bytes(text_src.read_bytes())

    id_map_dir = ROOT / "data" / "external_transformed"
    id_src = id_map_dir / f"{split}_id_map.csv"
    id_dst = id_map_dir / f"{new_split}_id_map.csv"
    if id_src.exists():
        id_dst.write_bytes(id_src.read_bytes())


def _build_split(split: str, args: argparse.Namespace, config: dict, model, resize, transform, device) -> dict:
    cover_dir = _resolve(args.cover_dir)
    image_dir = ROOT / "src_2" / "embedding" / "image"
    src_path = image_dir / f"image_embeddings_{split}.parquet"
    new_split = f"{split}_{args.suffix}"
    dst_path = image_dir / f"image_embeddings_{new_split}.parquet"
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    df = pd.read_parquet(src_path)
    if args.limit:
        work_df = df.head(args.limit).copy()
    else:
        work_df = df

    items: list[tuple[int, list[Image.Image]]] = []
    missing = 0
    fallback = 0
    yolo_cfg = config["yolo_detection"]
    mapped_cover_paths = _cover_path_map(split)
    for anime_id in tqdm(work_df["id"].astype(int).tolist(), desc=f"YOLO crops [{split}]"):
        path = mapped_cover_paths.get(anime_id, _cover_path(cover_dir, anime_id))
        img = load_image(str(path)) if path.exists() else None
        if img is None:
            missing += 1
            continue
        crops = _get_crops(img, yolo_cfg)
        if len(crops) == 1 and crops[0].size == img.size:
            fallback += 1
        items.append((anime_id, crops))

    emb_map = _embed_items(items, model, resize, transform, device, args.batch_size)
    out_df = df.copy()
    yolo_cols = [f"yolo_{i}" for i in range(1024)]
    for col in yolo_cols:
        if col not in out_df.columns:
            raise ValueError(f"Missing yolo column in {src_path}: {col}")
    for idx, anime_id in enumerate(out_df["id"].astype(int).tolist()):
        emb = emb_map.get(anime_id)
        if emb is None:
            continue
        out_df.loc[idx, yolo_cols] = emb.astype(np.float32)
        out_df.loc[idx, "has_yolo"] = 1
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dst_path, index=False)
    _write_split_copy(split, new_split)

    return {
        "split": split,
        "new_split": new_split,
        "rows": int(len(df)),
        "embedded_yolo_rows": int(len(emb_map)),
        "missing_cover_rows": int(missing),
        "fallback_full_image_rows": int(fallback),
        "output": dst_path.relative_to(ROOT).as_posix(),
    }


def main() -> None:
    args = _parse_args()
    config = _load_config(_resolve(args.config))
    model, resize, transform, device = _load_model(args.model_path, config)
    summaries = []
    for split in args.splits:
        summaries.append(_build_split(split, args, config, model, resize, transform, device))
    summary_path = ROOT / "data" / "external_transformed" / f"mal2025_yolo_embedding_summary_{args.suffix}.json"
    summary_text = json.dumps({"splits": summaries}, indent=2)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
