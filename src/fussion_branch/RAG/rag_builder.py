import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

from src.fussion_branch.RAG.sparse_encoder import SparseEncoder, parse_genres, parse_studios, parse_voice_actors, parse_source

_RAG_CONFIG_PATH = Path("src/fussion_branch/configs/rag_config.yaml")


def _load_rag_config() -> dict:
    with open(_RAG_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_emb_map(parquet_path: str, col_prefix: str) -> dict:
    """Load embedding parquet → {id: np.ndarray}. Returns {} if file missing."""
    p = Path(parquet_path)
    if not p.exists():
        return {}
    df    = pd.read_parquet(p)
    cols  = [c for c in df.columns if c.startswith(col_prefix)]
    return dict(zip(df["id"].astype(int), df[cols].values.astype(np.float32)))


def build_collection():
    cfg = _load_rag_config()
    collection_name = cfg["qdrant"]["collection_name"]
    encoder_path    = cfg["paths"]["encoder_path"]
    train_csv       = cfg["paths"]["train_csv"]
    text_emb_dir    = cfg["paths"]["text_emb_dir"]
    image_emb_dir   = cfg["paths"].get("image_emb_dir", "")
    text_emb_dim    = cfg["embedding"]["text_emb_dim"]
    image_emb_dim   = cfg["embedding"].get("image_emb_dim", 1024)
    batch_size      = cfg["indexing"]["batch_size"]

    df = pd.read_csv(train_csv)

    # Build sparse encoder from training data and save
    encoder = SparseEncoder().fit(df)
    encoder.save(encoder_path)
    print(f"Sparse vocab size: {encoder.dim}  (saved → {encoder_path})")

    # Load text embeddings (optional)
    text_emb_map = _load_emb_map(
        str(Path(text_emb_dir) / "text_embeddings_train.parquet"), "emb_"
    )
    if text_emb_map:
        print(f"Text embeddings loaded: {len(text_emb_map)} entries")
    else:
        print(f"Text embeddings not found → sparse-only mode")

    # Load image embeddings (optional)
    image_emb_map = _load_emb_map(
        str(Path(image_emb_dir) / "image_embeddings_train.parquet"), "img_"
    ) if image_emb_dir else {}
    if image_emb_map:
        print(f"Image embeddings loaded: {len(image_emb_map)} entries → image dense vector enabled")
    else:
        print(f"Image embeddings not found → image vector disabled")

    # Connect to Qdrant server
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    vectors_config: dict = {}
    if text_emb_map:
        vectors_config["text"]  = VectorParams(size=text_emb_dim,  distance=Distance.COSINE)
    if image_emb_map:
        vectors_config["image"] = VectorParams(size=image_emb_dim, distance=Distance.COSINE)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config={"genre_studio": models.SparseVectorParams()},
    )

    # Payload indexes for fast server-side filtering
    for field, schema in [
        ("release_year",    models.PayloadSchemaType.INTEGER),
        ("release_quarter", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema,
        )

    points = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing training set"):
        genres        = parse_genres(row["genres"])
        studios       = parse_studios(row["studios"])
        voice_actors  = parse_voice_actors(row["voice_actor_names"])
        source        = parse_source(row["source"])
        indices, values = encoder.encode(genres, studios, voice_actors, source)

        if not indices:
            skipped += 1
            continue

        payload = row.to_dict()
        payload["genres_parsed"]       = genres
        payload["studios_parsed"]      = studios
        payload["voice_actors_parsed"] = voice_actors

        anime_id = int(row["id"])
        vector: dict = {
            "genre_studio": models.SparseVector(indices=indices, values=values)
        }
        if text_emb_map and anime_id in text_emb_map:
            vector["text"]  = text_emb_map[anime_id].tolist()
        if image_emb_map and anime_id in image_emb_map:
            vector["image"] = image_emb_map[anime_id].tolist()

        points.append(
            models.PointStruct(id=anime_id, vector=vector, payload=payload)
        )

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points.clear()

    if points:
        client.upsert(collection_name=collection_name, points=points)

    count = client.count(collection_name).count
    modalities = ["sparse(genre/studio/voice/source)"]
    if text_emb_map:  modalities.append("dense-text(384)")
    if image_emb_map: modalities.append("dense-image(1024)")
    print(f"Collection '{collection_name}': {count} points indexed  (skipped={skipped})")
    print(f"  Retrieval mode: {' + '.join(modalities)}")
