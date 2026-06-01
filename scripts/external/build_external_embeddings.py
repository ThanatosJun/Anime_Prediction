"""
Build text and image embeddings for src_2 external splits.

Inputs are created by prepare_external_model_inputs.py:
- src_2/data/dataset/fusion_meta_clean_<split>_v2.csv
- data/external_transformed/<split>_id_map.csv

Outputs:
- src_2/embedding/text/text_embeddings_<split>.parquet
- src_2/embedding/image/image_embeddings_<split>.parquet
- src_2/embedding/image_rag/image_embeddings_<split>.parquet
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ["mal2025_popularity_local_ready", "mal2025_dual_local_ready"]
EMB_DIM_IMAGE = 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external split embeddings.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--modality", choices=["text", "image", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=32, help="Image batch size.")
    parser.add_argument(
        "--image-model-path",
        default=None,
        help="Optional Swin model directory. Defaults to src_2/component_image/image_encoder_config.yaml.",
    )
    return parser.parse_args()


def _load_text_embedder():
    text_dir = ROOT / "src_2" / "component_text"
    if str(text_dir) not in sys.path:
        sys.path.insert(0, str(text_dir))
    from output import TextEmbedder

    return TextEmbedder()


def _load_image_components(image_model_path: str | None = None):
    image_dir = ROOT / "src_2" / "component_image"
    if str(image_dir) not in sys.path:
        sys.path.insert(0, str(image_dir))
    sys.modules.pop("config", None)

    spec = importlib.util.spec_from_file_location("external_image_model", image_dir / "model.py")
    model_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(model_mod)

    proc_spec = importlib.util.spec_from_file_location("external_image_process", image_dir / "image_process.py")
    proc_mod = importlib.util.module_from_spec(proc_spec)
    assert proc_spec.loader is not None
    proc_spec.loader.exec_module(proc_mod)

    cfg_spec = importlib.util.spec_from_file_location("external_image_config", image_dir / "config.py")
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    assert cfg_spec.loader is not None
    cfg_spec.loader.exec_module(cfg_mod)

    cfg = cfg_mod.load_config(str(image_dir / "image_encoder_config.yaml"))
    model_path = Path(image_model_path) if image_model_path else Path(cfg["inference"]["model_path"])
    if not model_path.is_absolute() and not model_path.exists():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Image model not found: {model_path}. Provide the Swin model before building image embeddings."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_mod.load_model({"model": {"name": str(model_path)}}).to(device).eval()
    resize = proc_mod.ResizeWithPad(int(cfg["data"].get("image_size", 224)))
    transform = proc_mod.get_transform_original(int(cfg["data"].get("image_size", 224)))
    return model_mod, proc_mod, model, resize, transform, device


def _model_input_path(split: str) -> Path:
    return ROOT / "src_2" / "data" / "dataset" / f"fusion_meta_clean_{split}_v2.csv"


def _id_map_path(split: str) -> Path:
    return ROOT / "data" / "external_transformed" / f"{split}_id_map.csv"


def build_text_embeddings(split: str) -> Path:
    df = pd.read_csv(_model_input_path(split))
    embedder = _load_text_embedder()
    id_to_emb = embedder.embed_dataframe(df, text_col="description", id_col="id")

    emb_cols = [f"emb_{i:03d}" for i in range(embedder.embedding_dim)]
    rows = []
    for anime_id, emb in id_to_emb.items():
        row = {"id": anime_id}
        row.update(dict(zip(emb_cols, emb.tolist())))
        rows.append(row)

    out_dir = ROOT / "src_2" / "embedding" / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"text_embeddings_{split}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path


def _zero_image() -> list[float]:
    return [0.0] * EMB_DIM_IMAGE


def build_image_embeddings(split: str, batch_size: int, image_model_path: str | None = None) -> tuple[Path, Path]:
    id_map = pd.read_csv(_id_map_path(split))
    model_mod, proc_mod, model, resize, transform, device = _load_image_components(image_model_path)

    cover_cols = [f"cover_{i}" for i in range(EMB_DIM_IMAGE)]
    banner_cols = [f"banner_{i}" for i in range(EMB_DIM_IMAGE)]
    yolo_cols = [f"yolo_{i}" for i in range(EMB_DIM_IMAGE)]
    rag_cols = [f"img_{i}" for i in range(EMB_DIM_IMAGE)]

    rows_full = []
    rows_rag = []
    pending = []

    def flush() -> None:
        if not pending:
            return
        ids, images = zip(*pending)
        batch = torch.stack([transform(resize(img)) for img in images]).to(device)
        with torch.no_grad():
            embs = model_mod.get_embedding(model, batch).cpu().numpy()
        for anime_id, emb in zip(ids, embs):
            emb_list = emb.astype(np.float32).tolist()
            full = {"id": anime_id}
            full.update(dict(zip(yolo_cols, _zero_image())))
            full["has_yolo"] = 0
            full.update(dict(zip(cover_cols, emb_list)))
            full["has_cover"] = 1
            full.update(dict(zip(banner_cols, _zero_image())))
            full["has_banner"] = 0
            rows_full.append(full)

            rag = {"id": anime_id}
            rag.update(dict(zip(rag_cols, emb_list)))
            rows_rag.append(rag)
        pending.clear()

    for row in tqdm(id_map.itertuples(index=False), total=len(id_map), desc=f"image/{split}"):
        anime_id = int(row.id)
        path = Path(str(row.local_cover_image_path))
        if not path.is_absolute():
            path = ROOT / path
        image = proc_mod.load_image(str(path)) if path.exists() else None
        if image is None:
            full = {"id": anime_id}
            full.update(dict(zip(yolo_cols, _zero_image())))
            full["has_yolo"] = 0
            full.update(dict(zip(cover_cols, _zero_image())))
            full["has_cover"] = 0
            full.update(dict(zip(banner_cols, _zero_image())))
            full["has_banner"] = 0
            rows_full.append(full)
            continue
        pending.append((anime_id, image))
        if len(pending) >= batch_size:
            flush()
    flush()

    image_dir = ROOT / "src_2" / "embedding" / "image"
    image_rag_dir = ROOT / "src_2" / "embedding" / "image_rag"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_rag_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"image_embeddings_{split}.parquet"
    image_rag_path = image_rag_dir / f"image_embeddings_{split}.parquet"
    pd.DataFrame(rows_full).to_parquet(image_path, index=False)
    pd.DataFrame(rows_rag).to_parquet(image_rag_path, index=False)
    return image_path, image_rag_path


def main() -> None:
    args = _parse_args()
    for split in args.splits:
        if args.modality in ("text", "both"):
            out = build_text_embeddings(split)
            print(f"[text/{split}] -> {out.relative_to(ROOT)}")
        if args.modality in ("image", "both"):
            image_out, rag_out = build_image_embeddings(split, args.batch_size, args.image_model_path)
            print(f"[image/{split}] -> {image_out.relative_to(ROOT)}")
            print(f"[image-rag/{split}] -> {rag_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
