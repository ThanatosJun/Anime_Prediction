import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.fussion_branch.RAG.sparse_encoder import SparseEncoder, parse_genres, parse_studios

_RAG_CONFIG_PATH = Path("src/fussion_branch/configs/rag_config.yaml")


def _load_rag_config() -> dict:
    with open(_RAG_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_text_emb(split: str, text_emb_dir: str) -> dict:
    path = Path(text_emb_dir) / f"text_embeddings_{split}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return dict(zip(df["id"].astype(int), df[emb_cols].values.astype(np.float32)))


def query_split(
    client: QdrantClient,
    encoder: SparseEncoder,
    df: pd.DataFrame,
    split: str,
    text_emb_map: dict,
    fallback_popularity: float,
    fallback_score: float,
    collection_name: str,
    top_k: int,
    prefetch_k: int,
) -> pd.DataFrame:
    rows = []
    use_dense = len(text_emb_map) > 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"RAG query [{split}]"):
        genres   = parse_genres(row["genres"])
        studios  = parse_studios(row["studios"])
        indices, values = encoder.encode(genres, studios)
        anime_id = int(row["id"])

        rag_row = {
            "id":               anime_id,
            "rag_title_romaji": None,
            "rag_popularity":   fallback_popularity,
            "rag_score":        fallback_score,
            "rag_release_year": 0,
            "rag_studios":      json.dumps([]),
            "rag_found":        False,
        }

        if not indices:
            rows.append(rag_row)
            continue

        text_vec = text_emb_map.get(anime_id) if use_dense else None

        if text_vec is not None:
            # Hybrid: sparse + dense → RRF fusion
            results = client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(indices=indices, values=values),
                        using="genre_studio",
                        limit=prefetch_k,
                    ),
                    models.Prefetch(
                        query=text_vec.tolist(),
                        using="text",
                        limit=prefetch_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            ).points
        else:
            # Fallback: sparse only (text embeddings not available)
            results = client.query_points(
                collection_name=collection_name,
                query=models.SparseVector(indices=indices, values=values),
                using="genre_studio",
                limit=top_k,
            ).points

        # Python post-filter: remove same-period or later anime
        filtered = [
            r for r in results
            if r.payload.get("release_year", 9999) < row["release_year"]
        ]

        if filtered:
            top1 = filtered[0].payload
            rag_row.update({
                "rag_title_romaji": top1.get("title_romaji"),
                "rag_popularity":   top1.get("popularity", fallback_popularity),
                "rag_score":        top1.get("meanScore", fallback_score),
                "rag_release_year": top1.get("release_year", 0),
                "rag_studios":      json.dumps(top1.get("studios_parsed", [])),
                "rag_found":        True,
            })

        rows.append(rag_row)

    return pd.DataFrame(rows)


def query_all_splits(splits=("train", "val", "test")):
    cfg = _load_rag_config()
    collection_name = cfg["qdrant"]["collection_name"]
    db_path         = cfg["qdrant"]["db_path"]
    encoder_path    = cfg["paths"]["encoder_path"]
    train_csv       = cfg["paths"]["train_csv"]
    text_emb_dir    = cfg["paths"]["text_emb_dir"]
    out_dir         = Path(cfg["paths"]["out_dir"])
    top_k           = cfg["query"]["top_k"]
    prefetch_k      = cfg["query"]["prefetch_k"]

    encoder = SparseEncoder.load(encoder_path)
    client  = QdrantClient(path=db_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    fallback_popularity = float(train_df["popularity"].mean())
    fallback_score      = float(train_df["meanScore"].mean())

    for split in splits:
        df           = pd.read_csv(f"data/fussion/fusion_meta_clean_{split}.csv")
        text_emb_map = _load_text_emb(split, text_emb_dir)
        mode         = "hybrid (sparse+dense)" if text_emb_map else "sparse only"
        print(f"  [{split}] query mode: {mode}  ({len(text_emb_map)} text embeddings)")

        out_df   = query_split(
            client, encoder, df, split, text_emb_map,
            fallback_popularity, fallback_score,
            collection_name, top_k, prefetch_k,
        )
        out_path = out_dir / f"rag_features_{split}.parquet"
        out_df.to_parquet(out_path, index=False)

        found_rate = out_df["rag_found"].mean() * 100
        print(f"  [{split}] → {out_path}  |  found={found_rate:.1f}%  |  shape={out_df.shape}")
