"""
RAG query for a single anime (inference / pre-release prediction).

Usage
-----
from src.fussion_branch.RAG.rag_query_single import RagQuerySingle

rag = RagQuerySingle()

result = rag.query(
    genres=["Action", "Fantasy"],
    studios=["MAPPA"],
    release_year=2024,
    release_quarter=3,
    description="A young hero awakens a forbidden power...",  # optional
    anime_id=None,   # set if the anime is already in the Qdrant index (to exclude self)
)
print(result)
# {
#   "rag_title_romaji": "Shingeki no Kyojin",
#   "rag_popularity":   142301.0,
#   "rag_score":        84.3,
#   "rag_release_year": 2013,
#   "rag_studios":      '["Wit Studio"]',
#   "rag_found":        True,
# }

Notes
-----
- genres / studios can be raw strings (JSON or Python list literal) or already-parsed list[str].
- description is embedded on-the-fly via TextEmbedder (requires model download on first run).
- If description is None or too short, falls back to sparse-only search.
- anime_id should be set when the anime already exists in the Qdrant collection
  (e.g. querying a train/val/test anime) to prevent self-retrieval.
"""
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient, models

from src.fussion_branch.RAG.sparse_encoder import SparseEncoder, parse_genres, parse_studios, parse_voice_actors, parse_source
from src.fussion_branch.text_embedding import TextEmbedder

_RAG_CONFIG_PATH = Path("src/fussion_branch/configs/rag_config.yaml")


def _load_rag_config() -> dict:
    with open(_RAG_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _parse_input(value: Union[str, list]) -> list[str]:
    """Accept a pre-parsed list[str] or a raw string (JSON / Python literal)."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        import ast
        parsed = ast.literal_eval(value)
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []


def _build_time_filter(target_year: int, target_quarter, anime_id: Optional[int]) -> models.Filter:
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
    must_not = [models.HasIdCondition(has_id=[anime_id])] if anime_id is not None else []
    return models.Filter(must=[time_cond], must_not=must_not)


class RagQuerySingle:
    """
    Stateful RAG query client for single-anime inference.

    Loads encoder, Qdrant client, fallback stats, and TextEmbedder once,
    then serves repeated `query()` calls efficiently.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_rag_config() if config_path is None else (
            yaml.safe_load(open(config_path))
        )
        self._collection_name = cfg["qdrant"]["collection_name"]
        self._top_k           = cfg["query"]["top_k"]
        self._prefetch_k      = cfg["query"]["prefetch_k"]
        train_csv             = cfg["paths"]["train_csv"]

        self._encoder = SparseEncoder.load(cfg["paths"]["encoder_path"])
        self._client  = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
        self._embedder: Optional[TextEmbedder] = None  # lazy init

        # fallback values from training distribution
        train_df = pd.read_csv(train_csv)
        self._fallback_popularity = float(train_df["popularity"].mean())
        self._fallback_score      = float(train_df["meanScore"].mean())
        self._fallback_episodes   = float(train_df["episodes"].mean())

        # check whether the collection has a dense "text" vector namespace
        info = self._client.get_collection(self._collection_name)
        self._collection_has_dense = "text" in (info.config.params.vectors or {})

    def _get_embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = TextEmbedder()
        return self._embedder

    def query(
        self,
        genres: Union[str, list],
        studios: Union[str, list],
        release_year: int,
        release_quarter: Optional[int],
        description: Optional[str] = None,
        voice_actor_names: Optional[Union[str, list]] = None,
        source: Optional[str] = None,
        anime_id: Optional[int] = None,
    ) -> dict:
        """
        Query Qdrant for the best matching historical anime.

        Parameters
        ----------
        genres            : list[str] or raw string — e.g. ["Action", "Fantasy"]
        studios           : list[str] or raw string — e.g. ["MAPPA"]
        release_year      : target anime's release year
        release_quarter   : 1–4, or None if unknown
        description       : synopsis text (triggers dense search when provided)
        voice_actor_names : pipe-separated string or list[str] — e.g. "Hanae Natsuki|Kana Hanazawa"
        source            : source material type — e.g. "MANGA", "ORIGINAL", "GAME"
        anime_id          : AniList id to exclude from results (prevents self-match)

        Returns
        -------
        dict with keys:
            rag_title_romaji, rag_popularity, rag_score, rag_release_year,
            rag_episodes, rag_genres, rag_format, rag_studios, rag_found
        """
        fallback = {
            "rag_title_romaji": None,
            "rag_popularity":   self._fallback_popularity,
            "rag_score":        self._fallback_score,
            "rag_release_year": 0,
            "rag_episodes":     self._fallback_episodes,
            "rag_genres":       json.dumps([]),
            "rag_format":       None,
            "rag_studios":      json.dumps([]),
            "rag_found":        False,
        }

        # ── 1. sparse vector ─────────────────────────────────────────────────
        genres_list  = parse_genres(genres)   if isinstance(genres,  str) else _parse_input(genres)
        studios_list = parse_studios(studios) if isinstance(studios, str) else _parse_input(studios)
        va_list      = (
            parse_voice_actors(voice_actor_names) if isinstance(voice_actor_names, str)
            else (voice_actor_names or [])
        )
        src = parse_source(source) if source else ""
        indices, values = self._encoder.encode(genres_list, studios_list, va_list, src)

        if not indices:
            return fallback

        # ── 2. server-side filter: time + self-exclusion ──────────────────────
        query_filter = _build_time_filter(release_year, release_quarter, anime_id)

        # ── 3. dense vector (optional) ────────────────────────────────────────
        text_vec: Optional[np.ndarray] = None
        if description and self._collection_has_dense:
            emb = self._get_embedder().encode(description)
            if np.any(emb):
                text_vec = emb

        # ── 4. Qdrant query ───────────────────────────────────────────────────
        if text_vec is not None:
            results = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    models.Prefetch(
                        query=models.SparseVector(indices=indices, values=values),
                        using="genre_studio",
                        limit=self._prefetch_k,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=text_vec.tolist(),
                        using="text",
                        limit=self._prefetch_k,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=self._top_k,
            ).points
            mode = "hybrid"
        else:
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=models.SparseVector(indices=indices, values=values),
                using="genre_studio",
                query_filter=query_filter,
                limit=self._top_k,
            ).points
            mode = "sparse-only"

        if not results:
            return fallback

        # ── 5. extract top-1 result ───────────────────────────────────────────
        top1 = results[0].payload
        return {
            "rag_title_romaji": top1.get("title_romaji"),
            "rag_popularity":   top1.get("popularity",    self._fallback_popularity),
            "rag_score":        top1.get("meanScore",     self._fallback_score),
            "rag_release_year": top1.get("release_year",  0),
            "rag_episodes":     top1.get("episodes",      self._fallback_episodes),
            "rag_genres":       json.dumps(top1.get("genres_parsed", [])),
            "rag_format":       top1.get("format"),
            "rag_studios":      json.dumps(top1.get("studios_parsed", [])),
            "rag_found":        True,
            "_mode":            mode,
        }
