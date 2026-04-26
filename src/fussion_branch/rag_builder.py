import json
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.fussion_branch.sparse_encoder import SparseEncoder, parse_genres, parse_studios

COLLECTION_NAME = "anime_rag"
DB_PATH = "qdrant_db"
ENCODER_PATH = "artifacts/sparse_encoder.json"
TRAIN_CSV = "data/fussion/fusion_meta_train.csv"
BATCH_SIZE = 256


def build_collection():
    df = pd.read_csv(TRAIN_CSV)

    # Build sparse encoder from training data and save
    encoder = SparseEncoder().fit(df)
    encoder.save(ENCODER_PATH)
    print(f"Sparse vocab size: {encoder.dim}  (saved → {ENCODER_PATH})")

    # Init local Qdrant
    Path(DB_PATH).mkdir(exist_ok=True)
    client = QdrantClient(path=DB_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={},
        sparse_vectors_config={"genre_studio": models.SparseVectorParams()},
    )

    points = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing training set"):
        genres  = parse_genres(row["genres"])
        studios = parse_studios(row["studios"])
        indices, values = encoder.encode(genres, studios)

        if not indices:
            skipped += 1
            continue

        payload = row.to_dict()
        payload["genres_parsed"]  = genres
        payload["studios_parsed"] = studios

        points.append(
            models.PointStruct(
                id=int(row["id"]),
                vector={"genre_studio": models.SparseVector(indices=indices, values=values)},
                payload=payload,
            )
        )

        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points.clear()

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(COLLECTION_NAME).count
    print(f"Collection '{COLLECTION_NAME}': {count} points indexed  (skipped {skipped} with empty sparse vector)")
