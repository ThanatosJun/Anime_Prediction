import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.fussion_branch.RAG.sparse_encoder import SparseEncoder, parse_genres, parse_studios, parse_voice_actors, parse_source

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


def _build_time_filter(target_year: int, target_quarter, anime_id: int) -> models.Filter:
    """Server-side filter: only return anime released strictly before the target period."""
    try:
        tq = int(target_quarter)
        time_cond = models.Filter(
            should=[
                models.FieldCondition(key="release_year", range=models.Range(lt=target_year)),
                models.Filter(must=[
                    models.FieldCondition(key="release_year", match=models.MatchValue(value=target_year)),
                    models.FieldCondition(key="release_quarter", range=models.Range(lt=tq)),
                ]),
            ]
        )
    except (TypeError, ValueError):
        time_cond = models.Filter(
            must=[models.FieldCondition(key="release_year", range=models.Range(lt=target_year))]
        )
    return models.Filter(
        must=[time_cond],
        must_not=[models.HasIdCondition(has_id=[anime_id])],
    )


def query_split(
    client: QdrantClient,
    encoder: SparseEncoder,
    df: pd.DataFrame,
    split: str,
    text_emb_map: dict,
    fallback_popularity: float,
    fallback_score: float,
    fallback_episodes: float,
    collection_name: str,
    top_k: int,
    prefetch_k: int,
) -> pd.DataFrame:
    rows = []
    use_dense = len(text_emb_map) > 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"RAG query [{split}]"):
        genres       = parse_genres(row["genres"])
        studios      = parse_studios(row["studios"])
        voice_actors = parse_voice_actors(row["voice_actor_names"])
        source       = parse_source(row["source"])
        indices, values = encoder.encode(genres, studios, voice_actors, source)
        anime_id = int(row["id"])

        rag_row = {
            "id":               anime_id,
            "rag_title_romaji": None,
            "rag_popularity":   fallback_popularity,
            "rag_score":        fallback_score,
            "rag_release_year": 0,
            "rag_episodes":     fallback_episodes,
            "rag_genres":       json.dumps([]),
            "rag_format":       None,
            "rag_studios":      json.dumps([]),
            "rag_found":        False,
        }

        if not indices:
            rows.append(rag_row)
            continue

        query_filter = _build_time_filter(
            int(row["release_year"]), row["release_quarter"], anime_id
        )
        text_vec = text_emb_map.get(anime_id) if use_dense else None

        if text_vec is not None:
            results = client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(indices=indices, values=values),
                        using="genre_studio",
                        limit=prefetch_k,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=text_vec.tolist(),
                        using="text",
                        limit=prefetch_k,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=top_k,
            ).points
        else:
            results = client.query_points(
                collection_name=collection_name,
                query=models.SparseVector(indices=indices, values=values),
                using="genre_studio",
                query_filter=query_filter,
                limit=top_k,
            ).points

        if results:
            top1 = results[0].payload
            rag_row.update({
                "rag_title_romaji": top1.get("title_romaji"),
                "rag_popularity":   top1.get("popularity",   fallback_popularity),
                "rag_score":        top1.get("meanScore",    fallback_score),
                "rag_release_year": top1.get("release_year", 0),
                "rag_episodes":     top1.get("episodes",     fallback_episodes),
                "rag_genres":       json.dumps(top1.get("genres_parsed", [])),
                "rag_format":       top1.get("format"),
                "rag_studios":      json.dumps(top1.get("studios_parsed", [])),
                "rag_found":        True,
            })

        rows.append(rag_row)

    return pd.DataFrame(rows)


def query_all_splits(splits=("train", "val", "test")):
    cfg = _load_rag_config()
    collection_name = cfg["qdrant"]["collection_name"]
    encoder_path    = cfg["paths"]["encoder_path"]
    train_csv       = cfg["paths"]["train_csv"]
    text_emb_dir    = cfg["paths"]["text_emb_dir"]
    out_dir         = Path(cfg["paths"]["out_dir"])
    top_k           = cfg["query"]["top_k"]
    prefetch_k      = cfg["query"]["prefetch_k"]

    encoder = SparseEncoder.load(encoder_path)
    client  = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    fallback_popularity = float(train_df["popularity"].mean())
    fallback_score      = float(train_df["meanScore"].mean())
    fallback_episodes   = float(train_df["episodes"].mean())

    meta_dir = Path(cfg["paths"].get("meta_dir", "data/fussion"))
    for split in splits:
        df           = pd.read_csv(meta_dir / f"fusion_meta_clean_{split}.csv")
        text_emb_map = _load_text_emb(split, text_emb_dir)
        mode         = "hybrid (sparse+dense)" if text_emb_map else "sparse only"
        print(f"  [{split}] query mode: {mode}  ({len(text_emb_map)} text embeddings)")

        out_df   = query_split(
            client, encoder, df, split, text_emb_map,
            fallback_popularity, fallback_score, fallback_episodes,
            collection_name, top_k, prefetch_k,
        )
        out_path = out_dir / f"rag_features_{split}.parquet"
        out_df.to_parquet(out_path, index=False)

        found_rate = out_df["rag_found"].mean() * 100
        print(f"  [{split}] → {out_path}  |  found={found_rate:.1f}%  |  shape={out_df.shape}")
