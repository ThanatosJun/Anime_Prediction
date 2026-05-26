from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import requests
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, ResNetModel


TEXT_DIM = 768
PORTRAIT_DIM = 49


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build C1 Armenta Figure-2 character description and portrait features."
    )
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
        help="Reference baseline YAML config.",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--text-model", default="gpt2")
    parser.add_argument("--image-model", default="microsoft/resnet-50")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--portrait-batch-size", type=int, default=32)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-characters", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-portraits", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    data_cfg = config["data"]
    splits = args.splits or data_cfg.get("splits", ["train", "val", "test"])
    output_dir = Path(data_cfg["c1_character_emb_dir"])
    image_cache_dir = Path(data_cfg.get("c1_character_image_dir", ".exp/baseline/character_images"))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=args.local_files_only)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=args.local_files_only).to(device)
    text_model.eval()
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    if int(getattr(text_model.config, "hidden_size", TEXT_DIM)) != TEXT_DIM:
        raise ValueError(f"Expected GPT-2 hidden size {TEXT_DIM}; got {text_model.config.hidden_size}")

    image_processor = None
    image_model = None
    if not args.skip_portraits:
        image_processor = AutoImageProcessor.from_pretrained(
            args.image_model,
            local_files_only=args.local_files_only,
        )
        image_model = ResNetModel.from_pretrained(
            args.image_model,
            local_files_only=args.local_files_only,
        ).to(device)
        image_model.eval()

    raw = pd.read_csv(
        data_cfg.get("raw_anilist_path", "data/raw/anilist_anime_data_complete.csv"),
        usecols=[data_cfg.get("id_col", "id"), "characters"],
    )
    raw_map = dict(zip(raw[data_cfg.get("id_col", "id")].astype(int), raw["characters"]))

    for split in splits:
        meta_path = Path(data_cfg["meta_dir"]) / f"fusion_meta_clean_{split}.csv"
        df = pd.read_csv(meta_path, usecols=[data_cfg.get("id_col", "id")])
        if args.limit is not None:
            df = df.head(args.limit)
        ids = df[data_cfg.get("id_col", "id")].astype(int).tolist()
        records = [_extract_character_record(raw_map.get(anime_id), args.max_characters) for anime_id in ids]

        text_embeddings = _encode_texts(
            texts=[record["text"] for record in records],
            tokenizer=text_tokenizer,
            model=text_model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        portrait_embeddings = np.zeros((len(ids), PORTRAIT_DIM), dtype=np.float32)
        if image_model is not None and image_processor is not None:
            portrait_embeddings = _encode_portraits(
                records=records,
                image_processor=image_processor,
                image_model=image_model,
                image_cache_dir=image_cache_dir,
                device=device,
                timeout=args.timeout,
                batch_size=args.portrait_batch_size,
                download_workers=args.download_workers,
            )

        table = _build_table(ids, text_embeddings, portrait_embeddings)
        output_path = output_dir / f"c1_character_features_{split}.parquet"
        table.to_parquet(output_path, index=False)
        summary = _coverage_summary(records, portrait_embeddings)
        print(f"[{split}] saved {output_path} shape={table.shape} coverage={summary}")


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _extract_character_record(raw_value, max_characters: int) -> Dict[str, object]:
    chars = _parse_characters(raw_value)
    main = [item for item in chars if str(item.get("role", "")).upper() == "MAIN"]
    selected = (main or chars)[:max_characters]
    descriptions: List[str] = []
    urls: List[str] = []
    names: List[str] = []
    for item in selected:
        node = item.get("node") or {}
        name = ((node.get("name") or {}).get("full") or "").strip()
        description = _clean_text((node.get("description") or "").strip())
        image = node.get("image") or {}
        url = (image.get("medium") or image.get("large") or "").strip()
        if name:
            names.append(name)
        if description:
            descriptions.append(description)
        if url:
            urls.append(url)
    text = " ".join(descriptions) if descriptions else " ".join(names)
    return {
        "text": text if text else "[no character description]",
        "urls": urls,
        "has_description": bool(descriptions),
        "has_portrait_url": bool(urls),
    }


def _parse_characters(raw_value) -> List[dict]:
    if not isinstance(raw_value, str) or not raw_value.strip() or raw_value.strip() == "[]":
        return []
    try:
        parsed = json.loads(raw_value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\([^)]*source[^)]*\)", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\[[^]]*written[^]]*\]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="c1-char-gpt2"):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _encode_portraits(
    records: Sequence[Dict[str, object]],
    image_processor,
    image_model,
    image_cache_dir: Path,
    device: torch.device,
    timeout: float,
    batch_size: int,
    download_workers: int,
) -> np.ndarray:
    outputs = np.zeros((len(records), PORTRAIT_DIM), dtype=np.float32)
    indexed_records = list(enumerate(records))
    for start in tqdm(range(0, len(indexed_records), batch_size), desc="c1-char-resnet"):
        batch = indexed_records[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=max(1, int(download_workers))) as executor:
            images = list(
                executor.map(
                    lambda item: (
                        item[0],
                        _load_concatenated_portrait(item[1]["urls"], image_cache_dir, timeout),
                    ),
                    batch,
                )
            )
        images = [(row_idx, image) for row_idx, image in images if image is not None]
        if not images:
            continue
        row_indices = [row_idx for row_idx, _ in images]
        pil_images = [image for _, image in images]
        inputs = image_processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            hidden = image_model(**inputs).last_hidden_state[:, 0, :]
        vectors = hidden.reshape(hidden.shape[0], -1).detach().cpu().numpy().astype(np.float32)
        if vectors.shape[1] != PORTRAIT_DIM:
            raise ValueError(f"Expected portrait dim {PORTRAIT_DIM}, got {vectors.shape[1]}")
        for row_idx, vector in zip(row_indices, vectors):
            outputs[row_idx] = vector
    return outputs


def _load_concatenated_portrait(
    urls: Iterable[str],
    image_cache_dir: Path,
    timeout: float,
) -> Image.Image | None:
    images: List[Image.Image] = []
    for url in urls:
        image = _load_one_image(url, image_cache_dir, timeout)
        if image is not None:
            images.append(image)
    if not images:
        return None
    widths, heights = zip(*(image.size for image in images))
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(0, 0, 0))
    x_offset = 0
    for image in images:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def _load_one_image(url: str, image_cache_dir: Path, timeout: float) -> Image.Image | None:
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    name = hashlib.sha1(url.encode("utf-8")).hexdigest() + suffix
    path = image_cache_dir / name
    if not path.exists():
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            path.write_bytes(response.content)
        except Exception:
            return None
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _build_table(ids: Sequence[int], text_embeddings: np.ndarray, portrait_embeddings: np.ndarray) -> pd.DataFrame:
    text_cols = [f"char_gpt2_{idx:03d}" for idx in range(text_embeddings.shape[1])]
    portrait_cols = [f"char_resnet_{idx:03d}" for idx in range(portrait_embeddings.shape[1])]
    table = pd.DataFrame(np.concatenate([text_embeddings, portrait_embeddings], axis=1), columns=text_cols + portrait_cols)
    table.insert(0, "id", np.asarray(ids, dtype=np.int64))
    return table


def _coverage_summary(records: Sequence[Dict[str, object]], portrait_embeddings: np.ndarray) -> dict:
    return {
        "has_description": int(sum(bool(record["has_description"]) for record in records)),
        "has_portrait_url": int(sum(bool(record["has_portrait_url"]) for record in records)),
        "encoded_portrait": int(np.sum(np.linalg.norm(portrait_embeddings, axis=1) > 0)),
        "total": len(records),
    }


if __name__ == "__main__":
    main()
