from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_source_faithful_npz(
    config: dict,
    dataset_dir: Path,
    top_k: int,
    device: str = "cpu",
) -> None:
    """
    Build source-faithful SKAPP-style tensors from project anime inputs.

    Keeps source preprocessing/retrieval shape:
    - image_to_text (BLIP)
    - cls_vec / mean_pooling_vec (ViT)
    - merged_text_vec (AnglE/BERT-like)
    - nouns/verbs/adjectives (spaCy POS)
    - retrieval by sparse metadata/list intersections
    - retrieved visual/text/label tensors
    """

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    work_dir = Path(config["data"].get("skapp_full_dataset_dir", ".exp/baseline/skapp_full/dataset")).parent / "source_faithful"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]
    meta_dir = Path(data_cfg["meta_dir"])
    meta_suffix = data_cfg.get("meta_suffix", "")
    split_order = data_cfg.get("splits", ["train", "val", "test"])

    image_root = Path(data_cfg.get("image_dir", "data/image"))
    split_df = _load_or_build_preprocess(
        split_order=split_order,
        meta_dir=meta_dir,
        meta_suffix=meta_suffix,
        work_dir=work_dir,
        image_root=image_root,
        device=device,
    )
    split_df = _run_retrieval(
        split_df,
        retrieval_num=top_k,
        pool_mode="train_val",
        work_dir=work_dir,
    )

    for split, df in split_df.items():
        out_npz = dataset_dir / f"{split}.npz"
        _write_split_npz(df, out_npz=out_npz, top_k=top_k)


def _load_or_build_preprocess(
    split_order: list[str],
    meta_dir: Path,
    meta_suffix: str,
    work_dir: Path,
    image_root: Path,
    device: str,
) -> dict[str, pd.DataFrame]:
    split_df: dict[str, pd.DataFrame] = {}
    can_reuse = True
    required_cols = {
        "image_to_text",
        "mean_pooling_vec",
        "cls_vec",
        "merged_text",
        "merged_text_vec",
        "nouns",
        "verbs",
        "adjectives",
    }
    for split in split_order:
        pkl = work_dir / f"{split}.pkl"
        if not pkl.exists():
            can_reuse = False
            break
        df = pd.read_pickle(pkl)
        if not required_cols.issubset(set(df.columns)):
            can_reuse = False
            break
        split_df[split] = df
    if can_reuse:
        return split_df

    split_df = {}
    for split in split_order:
        path = meta_dir / f"fusion_meta_clean_{split}{meta_suffix}.csv"
        df = pd.read_csv(path)
        split_df[split] = _base_dataframe(df)
    return _run_preprocess(split_df, image_root=image_root, work_dir=work_dir, device=device)


def _base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(np.int64)
    out["image_id"] = out["id"]
    out["text"] = df.get("description", "").fillna("").astype(str)
    out["format"] = df.get("format", "").fillna("").astype(str)
    out["source"] = df.get("source", "").fillna("").astype(str)
    out["countryOfOrigin"] = df.get("countryOfOrigin", "").fillna("").astype(str)
    out["release_year"] = pd.to_numeric(df.get("release_year", 0), errors="coerce").fillna(0).astype(np.int64)
    out["release_quarter"] = pd.to_numeric(df.get("release_quarter", 0), errors="coerce").fillna(0).astype(np.int64)
    out["genres"] = _parse_list_col(df.get("genres", "[]"))
    out["studios"] = _parse_studios(df.get("studios", "[]"))
    out["voice_actor_names"] = _parse_pipe(df.get("voice_actor_names", ""))
    out["popularity"] = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0).astype(np.float32)
    out["meanScore"] = pd.to_numeric(df.get("meanScore", 0), errors="coerce").fillna(0).astype(np.float32)
    return out


def _run_preprocess(
    split_df: dict[str, pd.DataFrame],
    image_root: Path,
    work_dir: Path,
    device: str,
) -> dict[str, pd.DataFrame]:
    all_ids = np.concatenate([split_df[s]["id"].to_numpy() for s in ("train", "val", "test")], axis=0)
    id_to_image = {int(anime_id): _resolve_image_path(image_root, int(anime_id)) for anime_id in np.unique(all_ids)}

    # Load heavy models once.
    blip_processor, blip_model = _load_blip(device)
    vit_processor, vit_model = _load_vit(device)
    text_encoder = _load_text_encoder(device)
    nlp = _load_spacy()

    for split, df in split_df.items():
        image_to_text = []
        mean_vec = []
        cls_vec = []
        merged_text = []
        merged_vec = []
        nouns = []
        verbs = []
        adjectives = []

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"[source-preprocess] {split}"):
            image_path = id_to_image.get(int(row.id))
            caption = _image_to_text(image_path, blip_processor, blip_model, device)
            mean_v, cls_v = _image_to_vec(image_path, vit_processor, vit_model, device)
            merged = f"{row.text} {caption}".strip()
            text_v = _encode_text(merged, text_encoder)
            nn, vb, adj = _extract_pos(merged, nlp)

            image_to_text.append(caption)
            mean_vec.append(mean_v)
            cls_vec.append(cls_v)
            merged_text.append(merged)
            merged_vec.append(text_v)
            nouns.append(nn)
            verbs.append(vb)
            adjectives.append(adj)

        df["image_to_text"] = image_to_text
        df["mean_pooling_vec"] = mean_vec
        df["cls_vec"] = cls_vec
        df["merged_text"] = merged_text
        df["merged_text_vec"] = merged_vec
        df["nouns"] = nouns
        df["verbs"] = verbs
        df["adjectives"] = adjectives
        split_df[split] = df
        df.to_pickle(work_dir / f"{split}.pkl")

    return split_df


def _run_retrieval(
    split_df: dict[str, pd.DataFrame],
    retrieval_num: int,
    pool_mode: str = "train_val",
    work_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    if pool_mode == "train_val":
        retrieval_pool = pd.concat([split_df["train"], split_df["val"]], axis=0).reset_index(drop=True)
    elif pool_mode == "train":
        retrieval_pool = split_df["train"].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported pool_mode: {pool_mode}")

    all_features = [
        "format",
        "source",
        "countryOfOrigin",
        "release_year",
        "release_quarter",
        "genres",
        "studios",
        "voice_actor_names",
        "nouns",
        "verbs",
    ]
    list_features = {"genres", "studios", "voice_actor_names", "nouns", "verbs"}
    list_idx = [all_features.index(c) for c in list_features]

    pool_array = retrieval_pool[all_features].values
    N = len(retrieval_pool)

    for split, df in split_df.items():
        if _has_stacked_retrieval(df):
            split_df[split] = df
            continue

        df = df.copy()
        _ensure_retrieval_columns(df)
        start_idx = _first_incomplete_index(df)
        if start_idx >= len(df):
            split_df[split] = _stack_retrieved_feature(df, retrieval_pool)
            if work_dir is not None:
                split_df[split].to_pickle(work_dir / f"{split}.pkl")
            continue

        query_array = df[all_features].values
        for i in tqdm(range(start_idx, len(df)), desc=f"[source-retrieval] {split}"):
            sim = _calculate_similarity(query_array[i], pool_array, N=N, list_columns=list_idx)
            # Avoid self match if same id present in pool.
            same_id = retrieval_pool["id"].to_numpy() == int(df.iloc[i]["id"])
            sim[same_id] = -np.inf
            idx = np.argsort(sim)[::-1][:retrieval_num]
            candidates = retrieval_pool.iloc[idx]
            df.at[i, "retrieved_item_id"] = candidates["id"].astype(int).tolist()
            df.at[i, "retrieved_item_similarity"] = sim[idx].astype(float).tolist()
            df.at[i, "retrieved_label_popularity"] = candidates["popularity"].astype(float).tolist()
            df.at[i, "retrieved_label_meanScore"] = candidates["meanScore"].astype(float).tolist()
            if work_dir is not None and ((i - start_idx + 1) % 100 == 0):
                df.to_pickle(work_dir / f"{split}.pkl")

        if work_dir is not None:
            df.to_pickle(work_dir / f"{split}.pkl")
        split_df[split] = _stack_retrieved_feature(df, retrieval_pool)
        if work_dir is not None:
            split_df[split].to_pickle(work_dir / f"{split}.pkl")
    return split_df


def _stack_retrieved_feature(df_split: pd.DataFrame, df_database: pd.DataFrame) -> pd.DataFrame:
    index = {int(row.id): row for row in df_database.itertuples(index=False)}
    retrieved_cls = []
    retrieved_mean = []
    retrieved_text = []
    retrieved_label = []

    for ids, pop_list, score_list in zip(
        df_split["retrieved_item_id"],
        df_split["retrieved_label_popularity"],
        df_split["retrieved_label_meanScore"],
    ):
        cls_list = []
        mean_list = []
        text_list = []
        label_list = []
        for item_id, pop_value, score_value in zip(ids, pop_list, score_list):
            row = index.get(int(item_id))
            if row is None:
                continue
            cls_list.append(row.cls_vec)
            mean_list.append(row.mean_pooling_vec)
            text_list.append(row.merged_text_vec)
            label_list.append([float(pop_value), float(score_value)])
        retrieved_cls.append(cls_list)
        retrieved_mean.append(mean_list)
        retrieved_text.append(text_list)
        retrieved_label.append(label_list)

    out = df_split.copy()
    out["retrieved_visual_feature_embedding_cls"] = retrieved_cls
    out["retrieved_visual_feature_embedding_mean"] = retrieved_mean
    out["retrieved_textual_feature_embedding"] = retrieved_text
    out["retrieved_label_list"] = retrieved_label
    return out


def _write_split_npz(df: pd.DataFrame, out_npz: Path, top_k: int) -> None:
    ids = df["id"].to_numpy(dtype=np.int64)
    n = len(df)
    d_text = len(_as_vector(df.iloc[0]["merged_text_vec"])) if n else 768
    d_img = len(_as_vector(df.iloc[0]["cls_vec"])) if n else 768
    query_text = np.zeros((n, d_text), dtype=np.float32)
    query_image = np.zeros((n, d_img), dtype=np.float32)
    retrieved_text = np.zeros((n, top_k, d_text), dtype=np.float32)
    retrieved_image = np.zeros((n, top_k, d_img), dtype=np.float32)
    retrieved_labels = np.zeros((n, top_k, 2), dtype=np.float32)
    retrieved_mask = np.zeros((n, top_k), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        query_text[i] = _as_vector(row.merged_text_vec)
        query_image[i] = _as_vector(row.cls_vec)
        r_text = row.retrieved_textual_feature_embedding
        r_image = row.retrieved_visual_feature_embedding_cls
        r_label = row.retrieved_label_list
        k = min(top_k, len(r_text))
        for j in range(k):
            retrieved_text[i, j] = _as_vector(r_text[j])
            retrieved_image[i, j] = _as_vector(r_image[j])
            pop = float(r_label[j][0])
            score = float(r_label[j][1])
            retrieved_labels[i, j, 0] = np.log1p(max(pop, 0.0))
            retrieved_labels[i, j, 1] = score / 100.0
            retrieved_mask[i, j] = 1.0

    tmp_npz = out_npz.with_suffix(out_npz.suffix + ".tmp")
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(
            f,
            ids=ids,
            query_text=query_text,
            query_image=query_image,
            retrieved_text=retrieved_text,
            retrieved_image=retrieved_image,
            retrieved_labels=retrieved_labels,
            retrieved_mask=retrieved_mask,
            popularity=df["popularity"].to_numpy(dtype=np.float32),
            meanScore=df["meanScore"].to_numpy(dtype=np.float32),
            merged_text_vec=query_text,
            cls_vec=query_image,
            retrieved_visual_feature_embedding_cls=retrieved_image[:, :, None, :],
            retrieved_textual_feature_embedding=retrieved_text[:, :, None, :],
            retrieved_label_list_popularity=retrieved_labels[:, :, 0],
            retrieved_label_list_meanScore=retrieved_labels[:, :, 1],
        )
    _promote_tmp_npz(tmp_npz, out_npz)


def _promote_tmp_npz(tmp_npz: Path, out_npz: Path) -> None:
    for _ in range(6):
        try:
            if out_npz.exists():
                out_npz.unlink()
            tmp_npz.replace(out_npz)
            return
        except PermissionError:
            time.sleep(1.0)
    if out_npz.exists():
        out_npz.unlink()
    tmp_npz.replace(out_npz)


def _calculate_similarity(query_features, dataset_features, N: int, list_columns: Iterable[int]) -> np.ndarray:
    result = np.zeros((len(dataset_features), len(query_features)), dtype=np.int32)
    list_columns = set(list_columns)
    for i, feature in enumerate(query_features):
        if i in list_columns:
            fset = set(feature) if isinstance(feature, list) else set()
            result[:, i] = [
                1 if isinstance(other, list) and bool(fset.intersection(set(other))) else 0
                for other in dataset_features[:, i]
            ]
        else:
            result[:, i] = (dataset_features[:, i] == feature).astype(np.int32)
    n_values = result.sum(axis=0)
    score_weight = np.abs(np.log((N - n_values + 0.5) / (n_values + 0.5)))
    return np.dot(result, score_weight)


def _parse_list_col(series: pd.Series) -> list[list[str]]:
    out = []
    for item in series.fillna("[]"):
        if isinstance(item, list):
            values = item
        else:
            try:
                values = json.loads(item)
            except Exception:
                values = []
        out.append(sorted({str(v).strip() for v in values if str(v).strip()}))
    return out


def _parse_studios(series: pd.Series) -> list[list[str]]:
    out = []
    for item in series.fillna("[]"):
        names = []
        if isinstance(item, list):
            values = item
        else:
            try:
                values = json.loads(item)
            except Exception:
                values = []
        for v in values:
            if isinstance(v, dict):
                name = v.get("name")
                if name:
                    names.append(str(name).strip())
            elif v:
                names.append(str(v).strip())
        out.append(sorted({n for n in names if n}))
    return out


def _parse_pipe(series: pd.Series) -> list[list[str]]:
    out = []
    for item in series.fillna(""):
        tokens = [x.strip() for x in str(item).split("|") if x.strip()]
        out.append(sorted(set(tokens)))
    return out


def _resolve_image_path(image_root: Path, anime_id: int) -> Path | None:
    candidates = [
        image_root / f"{anime_id}_coverImage_extraLarge.jpg",
        image_root / f"{anime_id}_coverImage_medium.jpg",
        image_root / f"{anime_id}_coverImage_large.jpg",
        image_root / f"{anime_id}_bannerImage.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_blip(device: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def _load_vit(device: str):
    from transformers import ViTImageProcessor, ViTModel

    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def _load_text_encoder(device: str):
    try:
        from angle_emb import AnglE
    except Exception as exc:
        raise RuntimeError(
            "Source-faithful C3 requires `angle_emb` and "
            "`SeanLee97/angle-bert-base-uncased-nli-en-v1`."
        ) from exc

    model = AnglE.from_pretrained("SeanLee97/angle-bert-base-uncased-nli-en-v1", pooling_strategy="cls_avg")
    if device.startswith("cuda"):
        model = model.cuda()
    return ("angle", model)


def _load_spacy():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def _image_to_text(image_path: Path | None, processor, model, device: str) -> str:
    if image_path is None or not image_path.exists():
        return "0"
    from PIL import Image
    import torch

    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def _image_to_vec(image_path: Path | None, processor, model, device: str) -> tuple[list[float], list[float]]:
    if image_path is None or not image_path.exists():
        zeros = np.zeros(768, dtype=np.float32).tolist()
        return zeros, zeros
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_output = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy().astype(np.float32)
    mean_output = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return mean_output.tolist(), cls_output.tolist()


def _encode_text(text: str, encoder) -> list[float]:
    kind, model = encoder
    if kind == "angle":
        return _as_vector(model.encode(text, to_numpy=True)).tolist()
    return _as_vector(model.encode(text, convert_to_numpy=True, normalize_embeddings=False)).tolist()


def _extract_pos(text: str, nlp) -> tuple[list[str], list[str], list[str]]:
    doc = nlp(text)
    nouns = sorted({token.text for token in doc if token.pos_ == "NOUN"})
    verbs = sorted({token.text for token in doc if token.pos_ == "VERB"})
    adjectives = sorted({token.text for token in doc if token.pos_ == "ADJ"})
    return nouns, verbs, adjectives


def _as_vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr


def _has_retrieval_columns(df: pd.DataFrame) -> bool:
    needed = {
        "retrieved_item_id",
        "retrieved_item_similarity",
        "retrieved_label_popularity",
        "retrieved_label_meanScore",
        "retrieved_textual_feature_embedding",
        "retrieved_visual_feature_embedding_cls",
        "retrieved_label_list",
    }
    return needed.issubset(set(df.columns))


def _has_stacked_retrieval(df: pd.DataFrame) -> bool:
    needed = {
        "retrieved_visual_feature_embedding_cls",
        "retrieved_textual_feature_embedding",
        "retrieved_label_list",
    }
    return needed.issubset(set(df.columns))


def _ensure_retrieval_columns(df: pd.DataFrame) -> None:
    if "retrieved_item_id" not in df.columns:
        df["retrieved_item_id"] = [None] * len(df)
    if "retrieved_item_similarity" not in df.columns:
        df["retrieved_item_similarity"] = [None] * len(df)
    if "retrieved_label_popularity" not in df.columns:
        df["retrieved_label_popularity"] = [None] * len(df)
    if "retrieved_label_meanScore" not in df.columns:
        df["retrieved_label_meanScore"] = [None] * len(df)


def _first_incomplete_index(df: pd.DataFrame) -> int:
    for i in range(len(df)):
        if not isinstance(df.at[i, "retrieved_item_id"], list):
            return i
    return len(df)
