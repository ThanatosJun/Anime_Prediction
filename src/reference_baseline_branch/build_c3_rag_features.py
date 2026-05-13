from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yaml


RAG_MODES = ("none", "sparse", "dense", "hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build offline C3 RAG feature artifacts for reference baselines."
    )
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
        help="Reference baseline config YAML.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=RAG_MODES,
        default=list(RAG_MODES),
        help="RAG retrieval modes to generate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved train items to aggregate.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=None,
        help="Optional split subset to generate. Defaults to config data.splits.",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    builder = OfflineRagFeatureBuilder(config, top_k=args.top_k)
    for mode in dict.fromkeys(args.modes):
        print(f"[c3-rag] build mode={mode}")
        builder.build_mode(mode, splits=args.splits)


class OfflineRagFeatureBuilder:
    def __init__(self, config: dict, top_k: int):
        self.config = config
        self.data_cfg = config["data"]
        self.top_k = int(top_k)
        self.id_col = self.data_cfg.get("id_col", "id")
        self.meta = self._load_meta()
        self.out_root = Path(self.data_cfg.get("rag_features_root", ".exp/baseline/rag_features"))
        self.train_df = self.meta["train"].reset_index(drop=True)
        self.train_ids = self.train_df[self.id_col].astype(int).to_numpy()
        self.fallback = {
            "rag_popularity": float(pd.to_numeric(self.train_df["popularity"], errors="coerce").mean()),
            "rag_score": float(pd.to_numeric(self.train_df["meanScore"], errors="coerce").mean()),
            "rag_release_year": 0.0,
            "rag_episodes": float(pd.to_numeric(self.train_df["episodes"], errors="coerce").mean()),
            "rag_similarity_mean": 0.0,
            "rag_similarity_max": 0.0,
            "rag_topk_count": 0.0,
        }
        self.train_sparse_tokens = [set(_sparse_tokens(row)) for _, row in self.train_df.iterrows()]
        self.inverted_index = self._build_inverted_index(self.train_sparse_tokens)
        self.train_release_year = pd.to_numeric(self.train_df["release_year"], errors="coerce").fillna(0).astype(int).to_numpy()
        self.train_release_quarter = pd.to_numeric(self.train_df["release_quarter"], errors="coerce").fillna(0).astype(int).to_numpy()
        self.text_emb = self._load_text_embeddings()
        self.train_text_matrix = self.text_emb.get("train_matrix")
        self.train_text_ids = self.text_emb.get("train_ids")
        self.dense_score_cache: Dict[str, Dict[int, Dict[int, float]]] = {}

    def build_mode(self, mode: str, splits: Sequence[str] | None = None) -> None:
        if mode not in RAG_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        out_dir = self.out_root / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        split_names = list(splits) if splits is not None else list(self.meta)
        for split in split_names:
            df = self.meta[split]
            rows = [self._row_for_query(row, split, mode) for _, row in df.iterrows()]
            out_path = out_dir / f"rag_features_{split}.parquet"
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"  [{split}] {len(rows)} rows -> {out_path}")

    def _row_for_query(self, row: pd.Series, split: str, mode: str) -> dict:
        anime_id = int(row[self.id_col])
        if mode == "none":
            return self._empty_row(anime_id)

        candidates = self._retrieve(row, split, mode)
        if not candidates:
            return self._empty_row(anime_id)
        return self._aggregate_row(anime_id, candidates)

    def _retrieve(self, row: pd.Series, split: str, mode: str) -> List[tuple[int, float]]:
        allowed = self._allowed_train_indices(row, split)
        if not allowed:
            return []

        sparse_scores: Dict[int, float] = {}
        if mode in {"sparse", "hybrid"}:
            query_tokens = set(_sparse_tokens(row))
            sparse_scores = self._sparse_scores(query_tokens, allowed)

        dense_scores: Dict[int, float] = {}
        if mode in {"dense", "hybrid"}:
            dense_scores = self._dense_scores(row, split, allowed)

        if mode == "sparse":
            scores = sparse_scores
        elif mode == "dense":
            scores = dense_scores
        else:
            scores = self._rrf_scores(sparse_scores, dense_scores)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[: self.top_k]

    def _allowed_train_indices(self, row: pd.Series, split: str) -> set[int]:
        target_year = _safe_int(row.get("release_year"), default=0)
        target_quarter = _safe_int(row.get("release_quarter"), default=0)
        anime_id = _safe_int(row.get(self.id_col), default=-1)
        if target_year <= 0:
            mask = self.train_release_year > 0
        else:
            mask = (self.train_release_year < target_year) | (
                (self.train_release_year == target_year)
                & (self.train_release_quarter > 0)
                & (self.train_release_quarter < target_quarter)
            )
        if split == "train":
            mask = mask & (self.train_ids != anime_id)
        return set(np.where(mask)[0].tolist())

    def _sparse_scores(self, query_tokens: set[str], allowed: set[int]) -> Dict[int, float]:
        if not query_tokens:
            return {}
        counts: Counter[int] = Counter()
        for token in query_tokens:
            for idx in self.inverted_index.get(token, []):
                if idx in allowed:
                    counts[idx] += 1
        scores: Dict[int, float] = {}
        for idx, count in counts.items():
            denom = math.sqrt(max(len(query_tokens), 1) * max(len(self.train_sparse_tokens[idx]), 1))
            scores[idx] = float(count / denom)
        return scores

    def _dense_scores(self, row: pd.Series, split: str, allowed: set[int]) -> Dict[int, float]:
        if self.train_text_matrix is None:
            return {}
        self._prepare_dense_score_cache(split)
        return self.dense_score_cache.get(split, {}).get(int(row[self.id_col]), {})

    def _prepare_dense_score_cache(self, split: str, batch_size: int = 512) -> None:
        if split in self.dense_score_cache:
            return
        df = self.meta[split].reset_index(drop=True)
        emb_map = self.text_emb.get(split)
        if emb_map is None or self.train_text_matrix is None:
            self.dense_score_cache[split] = {}
            return

        dim = self.train_text_matrix.shape[1]
        query_matrix = np.zeros((len(df), dim), dtype=np.float32)
        valid = np.zeros(len(df), dtype=bool)
        for row_idx, row in df.iterrows():
            vec = emb_map.get(int(row[self.id_col]))
            if vec is not None:
                query_matrix[row_idx] = vec
                valid[row_idx] = True

        split_scores: Dict[int, Dict[int, float]] = {}
        top_n = max(self.top_k * 4, self.top_k)
        train_t = self.train_text_matrix.T
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            sims = query_matrix[start:end] @ train_t
            for local_idx, row_idx in enumerate(range(start, end)):
                if not valid[row_idx]:
                    continue
                row = df.iloc[row_idx]
                allowed = self._allowed_train_indices(row, split)
                if not allowed:
                    continue
                allowed_sorted = np.fromiter(sorted(allowed), dtype=np.int64)
                allowed_sims = sims[local_idx, allowed_sorted]
                local_top_n = min(top_n, len(allowed_sims))
                if local_top_n == 0:
                    continue
                top_local = np.argpartition(-allowed_sims, local_top_n - 1)[:local_top_n]
                split_scores[int(row[self.id_col])] = {
                    int(allowed_sorted[pos]): float(allowed_sims[pos])
                    for pos in top_local
                    if allowed_sims[pos] > 0
                }
        self.dense_score_cache[split] = split_scores

    def _query_text_vector(self, row: pd.Series, split: str) -> np.ndarray | None:
        emb_map = self.text_emb.get(split)
        if emb_map is None:
            return None
        vec = emb_map.get(int(row[self.id_col]))
        if vec is None:
            return None
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return (vec / norm).astype(np.float32)

    def _rrf_scores(
        self,
        sparse_scores: Dict[int, float],
        dense_scores: Dict[int, float],
        k: int = 60,
    ) -> Dict[int, float]:
        combined: Dict[int, float] = defaultdict(float)
        for scores in (sparse_scores, dense_scores):
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for rank, (idx, _) in enumerate(ranked, start=1):
                combined[idx] += 1.0 / (k + rank)
        return dict(combined)

    def _aggregate_row(self, anime_id: int, candidates: Sequence[tuple[int, float]]) -> dict:
        indices = [idx for idx, _ in candidates]
        scores = np.array([score for _, score in candidates], dtype=np.float64)
        df = self.train_df.iloc[indices]
        top = df.iloc[0]
        genres = sorted({token for value in df["genres"] for token in _parse_literal_list(value)})
        studios = sorted({token for value in df["studios"] for token in _parse_studios(value)})
        return {
            "id": anime_id,
            "rag_title_romaji": top.get("title_romaji"),
            "rag_popularity": float(pd.to_numeric(df["popularity"], errors="coerce").mean()),
            "rag_score": float(pd.to_numeric(df["meanScore"], errors="coerce").mean()),
            "rag_release_year": float(pd.to_numeric(df["release_year"], errors="coerce").mean()),
            "rag_episodes": float(pd.to_numeric(df["episodes"], errors="coerce").mean()),
            "rag_similarity_mean": float(scores.mean()),
            "rag_similarity_max": float(scores.max()),
            "rag_topk_count": float(len(indices)),
            "rag_genres": json.dumps(genres, ensure_ascii=False),
            "rag_format": top.get("format") if pd.notna(top.get("format")) else None,
            "rag_studios": json.dumps(studios, ensure_ascii=False),
            "rag_found": True,
        }

    def _empty_row(self, anime_id: int) -> dict:
        return {
            "id": anime_id,
            "rag_title_romaji": None,
            **self.fallback,
            "rag_genres": json.dumps([], ensure_ascii=False),
            "rag_format": None,
            "rag_studios": json.dumps([], ensure_ascii=False),
            "rag_found": False,
        }

    def _load_meta(self) -> Dict[str, pd.DataFrame]:
        meta_dir = Path(self.data_cfg["meta_dir"])
        return {
            split: pd.read_csv(meta_dir / f"fusion_meta_clean_{split}.csv")
            for split in self.data_cfg.get("splits", ["train", "val", "test"])
        }

    def _build_inverted_index(self, token_sets: Iterable[set[str]]) -> Dict[str, List[int]]:
        index: Dict[str, List[int]] = defaultdict(list)
        for row_idx, tokens in enumerate(token_sets):
            for token in tokens:
                index[token].append(row_idx)
        return dict(index)

    def _load_text_embeddings(self) -> dict:
        emb_dir = Path(self.data_cfg["text_emb_dir"])
        emb_cfg = self.config["features"]["text_embedding"]
        cache: dict = {}
        train_matrix = None
        train_ids = None
        for split in self.data_cfg.get("splits", ["train", "val", "test"]):
            path = emb_dir / emb_cfg["file_template"].format(split=split)
            if not path.exists():
                continue
            df = pd.read_parquet(path).set_index(self.id_col)
            cols = [col for col in df.columns if str(col).startswith(emb_cfg.get("prefix", ""))]
            values = df[cols].astype(np.float32)
            norms = np.linalg.norm(values.values, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = values.values / norms
            id_to_vec = {
                int(idx): normalized[row_idx]
                for row_idx, idx in enumerate(values.index.astype(int).tolist())
            }
            cache[split] = id_to_vec
            if split == "train":
                order = [int(item) for item in self.train_ids.tolist()]
                train_matrix = np.zeros((len(order), len(cols)), dtype=np.float32)
                for row_idx, anime_id in enumerate(order):
                    vec = id_to_vec.get(anime_id)
                    if vec is not None:
                        train_matrix[row_idx] = vec
                train_ids = np.array(order, dtype=np.int64)
        cache["train_matrix"] = train_matrix
        cache["train_ids"] = train_ids
        return cache


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sparse_tokens(row: pd.Series) -> List[str]:
    tokens: List[str] = []
    tokens.extend(f"genre:{value}" for value in _parse_literal_list(row.get("genres")))
    tokens.extend(f"studio:{value}" for value in _parse_studios(row.get("studios")))
    tokens.extend(f"voice:{value}" for value in _parse_pipe(row.get("voice_actor_names")))
    source = row.get("source")
    if pd.notna(source) and str(source).strip():
        tokens.append(f"source:{str(source).strip()}")
    return tokens


def _parse_literal_list(value) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _parse_studios(value) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    values: List[str] = []
    for item in parsed:
        if isinstance(item, dict):
            node = item.get("node", {})
            name = node.get("name")
            if name:
                values.append(str(name).strip())
    return values


def _parse_pipe(value) -> List[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _safe_int(value, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
