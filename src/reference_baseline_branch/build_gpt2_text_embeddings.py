from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "gpt2"
FEATURE_DIM = 768


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GPT-2 pooled synopsis embeddings for C1/C2 reference baselines."
    )
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
        help="Reference baseline YAML config.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the Hugging Face model to already exist in the local cache.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split smoke limit.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    data_cfg = config["data"]
    splits = args.splits or data_cfg.get("splits", ["train", "val", "test"])
    output_dir = Path(data_cfg["gpt2_text_emb_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = int(getattr(model.config, "hidden_size", FEATURE_DIM))
    if hidden_size != FEATURE_DIM:
        raise ValueError(
            f"C1/C2 GPT-2 reference feature expects hidden_size={FEATURE_DIM}, "
            f"but {args.model_name} has hidden_size={hidden_size}"
        )

    for split in splits:
        meta_suffix = data_cfg.get("meta_suffix", "")
        meta_path = Path(data_cfg["meta_dir"]) / f"fusion_meta_clean_{split}{meta_suffix}.csv"
        df = pd.read_csv(meta_path, usecols=[data_cfg.get("id_col", "id"), "title_romaji", "description"])
        if args.limit is not None:
            df = df.head(args.limit)

        texts = [_compose_text(row) for _, row in df.iterrows()]
        embeddings = _encode_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        table = _build_table(df[data_cfg.get("id_col", "id")].astype(np.int64).values, embeddings)
        output_path = output_dir / f"gpt2_text_embeddings_{split}.parquet"
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


def _compose_text(row: pd.Series) -> str:
    title = "" if pd.isna(row.get("title_romaji")) else str(row.get("title_romaji"))
    description = "" if pd.isna(row.get("description")) else str(row.get("description"))
    description = _clean_html(description)
    text = f"{title}. {description}".strip()
    return text if text else "[no synopsis]"


def _clean_html(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="gpt2"):
        batch_texts = texts[start : start + batch_size]
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


def _build_table(ids: np.ndarray, embeddings: np.ndarray) -> pd.DataFrame:
    columns = [f"gpt2_{idx:03d}" for idx in range(embeddings.shape[1])]
    table = pd.DataFrame(embeddings, columns=columns)
    table.insert(0, "id", ids)
    return table


if __name__ == "__main__":
    main()
