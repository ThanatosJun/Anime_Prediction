import json
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.fussion_branch.sparse_encoder import SparseEncoder, parse_genres, parse_studios

COLLECTION_NAME = "anime_rag"
DB_PATH = "qdrant_db"
ENCODER_PATH = "artifacts/sparse_encoder.json"
TOP_K = 10
OUT_DIR = Path("artifacts")


def query_split(client: QdrantClient, encoder: SparseEncoder,
                df: pd.DataFrame, split: str,
                fallback_popularity: float, fallback_score: float) -> pd.DataFrame:
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"RAG query [{split}]"):
        genres  = parse_genres(row["genres"])
        studios = parse_studios(row["studios"])
        indices, values = encoder.encode(genres, studios)

        rag_row = {
            "id":               int(row["id"]),
            "rag_title_romaji": None,
            "rag_popularity":   fallback_popularity,
            "rag_score":        fallback_score,
            "rag_release_year": 0,
            "rag_studios":      json.dumps([]),
            "rag_found":        False,
        }

        if indices:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=models.SparseVector(indices=indices, values=values),
                using="genre_studio",
                limit=TOP_K,
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
    encoder = SparseEncoder.load(ENCODER_PATH)
    client  = QdrantClient(path=DB_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv("data/fussion/fusion_meta_train.csv")
    fallback_popularity = float(train_df["popularity"].mean())
    fallback_score      = float(train_df["meanScore"].mean())

    for split in splits:
        df      = pd.read_csv(f"data/fussion/fusion_meta_{split}.csv")
        out_df  = query_split(client, encoder, df, split, fallback_popularity, fallback_score)
        out_path = OUT_DIR / f"rag_features_{split}.parquet"
        out_df.to_parquet(out_path, index=False)

        found_rate = out_df["rag_found"].mean() * 100
        print(f"  [{split}] → {out_path}  |  found={found_rate:.1f}%  |  shape={out_df.shape}")
