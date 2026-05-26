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
from sklearn.ensemble import GradientBoostingRegressor


RAG_MODES = (
    "none",
    "sparse",
    "dense",
    "hybrid",
    "selective",
    "skapp_proxy",
    "skapp_graph_proxy",
)


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
        self.image_emb = self._load_image_embeddings()
        self.train_image_matrix = self.image_emb.get("train_matrix")
        self.text_dim = 0 if self.train_text_matrix is None else int(self.train_text_matrix.shape[1])
        self.image_dim = 0 if self.train_image_matrix is None else int(self.train_image_matrix.shape[1])
        self.dense_score_cache: Dict[str, Dict[int, Dict[int, float]]] = {}
        self.contribution_model: GradientBoostingRegressor | None = None

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
        if mode == "skapp_proxy":
            candidates = self._retrieve_skapp_candidates(row, split)
            if not candidates:
                return self._empty_skapp_row(anime_id)
            return self._aggregate_skapp_row(anime_id, candidates)
        if mode == "skapp_graph_proxy":
            candidates = self._retrieve_skapp_candidates(row, split)
            if not candidates:
                return self._empty_skapp_graph_row(anime_id)
            return self._aggregate_skapp_graph_row(anime_id, candidates)

        candidates = self._retrieve(row, split, mode)
        if not candidates:
            return self._empty_row(anime_id)
        return self._aggregate_row(anime_id, candidates)

    def _retrieve(self, row: pd.Series, split: str, mode: str) -> List[tuple[int, float]]:
        allowed = self._allowed_train_indices(row, split)
        if not allowed:
            return []

        sparse_scores: Dict[int, float] = {}
        if mode in {"sparse", "hybrid", "selective"}:
            query_tokens = set(_sparse_tokens(row))
            sparse_scores = self._sparse_scores(query_tokens, allowed)

        dense_scores: Dict[int, float] = {}
        if mode in {"dense", "hybrid"}:
            dense_scores = self._dense_scores(row, split, allowed)

        if mode == "sparse":
            scores = sparse_scores
        elif mode == "dense":
            scores = dense_scores
        elif mode == "selective":
            return self._select_sparse_candidates(sparse_scores)
        else:
            scores = self._rrf_scores(sparse_scores, dense_scores)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[: self.top_k]

    def _retrieve_skapp_candidates(self, row: pd.Series, split: str) -> List[tuple[int, float]]:
        self._ensure_contribution_model()
        allowed = self._allowed_train_indices(row, split)
        if not allowed or self.contribution_model is None:
            return []

        sparse_scores = self._sparse_scores(set(_sparse_tokens(row)), allowed)
        dense_scores = self._dense_scores(row, split, allowed)
        rrf_scores = self._rrf_scores(sparse_scores, dense_scores)
        if not rrf_scores:
            return []

        pool_size = self.top_k
        pool = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:pool_size]
        features = np.asarray(
            [
                self._pair_features(
                    row=row,
                    candidate_idx=idx,
                    sparse_score=sparse_scores.get(idx, 0.0),
                    dense_score=dense_scores.get(idx, 0.0),
                    rrf_score=rrf_score,
                )
                for idx, rrf_score in pool
            ],
            dtype=np.float32,
        )
        contributions = np.asarray(self.contribution_model.predict(features), dtype=np.float64)
        threshold = float(np.median(contributions))
        selected = [
            (idx, float(score))
            for (idx, _), score in zip(pool, contributions)
            if score >= threshold
        ]
        if not selected:
            best_pos = int(np.argmax(contributions))
            selected = [(pool[best_pos][0], float(contributions[best_pos]))]
        selected = sorted(selected, key=lambda item: item[1], reverse=True)[: self.top_k]
        return selected

    def _ensure_contribution_model(self) -> None:
        if self.contribution_model is not None:
            return
        feature_rows: List[List[float]] = []
        labels: List[float] = []
        for _, row in self.train_df.iterrows():
            allowed = self._allowed_train_indices(row, "train")
            if not allowed:
                continue
            sparse_scores = self._sparse_scores(set(_sparse_tokens(row)), allowed)
            dense_scores = self._dense_scores(row, "train", allowed)
            rrf_scores = self._rrf_scores(sparse_scores, dense_scores)
            if not rrf_scores:
                continue
            pool_size = max(self.top_k * 4, self.top_k)
            pool = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:pool_size]
            for idx, rrf_score in pool:
                feature_rows.append(
                    self._pair_features(
                        row=row,
                        candidate_idx=idx,
                        sparse_score=sparse_scores.get(idx, 0.0),
                        dense_score=dense_scores.get(idx, 0.0),
                        rrf_score=rrf_score,
                    )
                )
                labels.append(self._pair_contribution_label(row, idx))

        if not feature_rows:
            self.contribution_model = None
            return
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
        model.fit(np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.float32))
        self.contribution_model = model
        print(f"  [skapp_proxy] trained contribution scorer on {len(labels)} train pairs")

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

    def _select_sparse_candidates(self, sparse_scores: Dict[int, float]) -> List[tuple[int, float]]:
        ranked = sorted(sparse_scores.items(), key=lambda item: item[1], reverse=True)
        top_candidates = ranked[: self.top_k]
        if not top_candidates:
            return []
        scores = np.array([score for _, score in top_candidates], dtype=np.float64)
        threshold = float(np.median(scores))
        selected = [(idx, score) for idx, score in top_candidates if score >= threshold]
        if selected:
            return selected
        return top_candidates[:1]

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

    def _aggregate_skapp_row(self, anime_id: int, candidates: Sequence[tuple[int, float]]) -> dict:
        indices = [idx for idx, _ in candidates]
        scores = np.array([score for _, score in candidates], dtype=np.float64)
        weights = _softmax(scores)
        df = self.train_df.iloc[indices]
        base = self._aggregate_row(anime_id, candidates)
        popularity = pd.to_numeric(df["popularity"], errors="coerce").fillna(self.fallback["rag_popularity"]).to_numpy()
        mean_score = pd.to_numeric(df["meanScore"], errors="coerce").fillna(self.fallback["rag_score"]).to_numpy()
        release_year = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).to_numpy()
        episodes = pd.to_numeric(df["episodes"], errors="coerce").fillna(self.fallback["rag_episodes"]).to_numpy()
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-12))))
        base.update(
            {
                "skapp_contribution_mean": float(scores.mean()),
                "skapp_contribution_max": float(scores.max()),
                "skapp_contribution_std": float(scores.std()) if len(scores) > 1 else 0.0,
                "skapp_attention_entropy": entropy,
                "skapp_selected_count": float(len(indices)),
                "skapp_weighted_popularity": float(np.sum(weights * popularity)),
                "skapp_weighted_score": float(np.sum(weights * mean_score)),
                "skapp_weighted_release_year": float(np.sum(weights * release_year)),
                "skapp_weighted_episodes": float(np.sum(weights * episodes)),
            }
        )
        return base

    def _aggregate_skapp_graph_row(self, anime_id: int, candidates: Sequence[tuple[int, float]]) -> dict:
        base = self._aggregate_skapp_row(anime_id, candidates)
        base.update(self._skapp_graph_features(candidates))
        return base

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

    def _empty_skapp_row(self, anime_id: int) -> dict:
        row = self._empty_row(anime_id)
        row.update(
            {
                "skapp_contribution_mean": 0.0,
                "skapp_contribution_max": 0.0,
                "skapp_contribution_std": 0.0,
                "skapp_attention_entropy": 0.0,
                "skapp_selected_count": 0.0,
                "skapp_weighted_popularity": self.fallback["rag_popularity"],
                "skapp_weighted_score": self.fallback["rag_score"],
                "skapp_weighted_release_year": 0.0,
                "skapp_weighted_episodes": self.fallback["rag_episodes"],
            }
        )
        return row

    def _empty_skapp_graph_row(self, anime_id: int) -> dict:
        row = self._empty_skapp_row(anime_id)
        row.update(self._skapp_graph_features([]))
        return row

    def _skapp_graph_features(self, candidates: Sequence[tuple[int, float]]) -> dict:
        mask = np.zeros(self.top_k, dtype=np.float32)
        contributions = np.zeros(self.top_k, dtype=np.float32)
        labels = np.zeros((self.top_k, 2), dtype=np.float32)
        text = np.zeros((self.top_k, self.text_dim), dtype=np.float32)
        image = np.zeros((self.top_k, self.image_dim), dtype=np.float32)

        for pos, (idx, score) in enumerate(candidates[: self.top_k]):
            candidate = self.train_df.iloc[idx]
            mask[pos] = 1.0
            contributions[pos] = float(score)
            labels[pos, 0] = math.log1p(max(_safe_float(candidate.get("popularity"), 0.0), 0.0))
            labels[pos, 1] = _safe_float(candidate.get("meanScore"), 0.0) / 100.0
            if self.train_text_matrix is not None:
                text[pos] = self.train_text_matrix[idx]
            if self.train_image_matrix is not None:
                image[pos] = self.train_image_matrix[idx]

        row: dict = {}
        for pos in range(self.top_k):
            row[f"skapp_graph_mask_{pos:02d}"] = float(mask[pos])
        for pos in range(self.top_k):
            row[f"skapp_graph_rrcp_{pos:02d}"] = float(contributions[pos])
        for pos in range(self.top_k):
            row[f"skapp_graph_label_pop_{pos:02d}"] = float(labels[pos, 0])
            row[f"skapp_graph_label_score_{pos:02d}"] = float(labels[pos, 1])
        for pos in range(self.top_k):
            for dim in range(self.text_dim):
                row[f"skapp_graph_text_{pos:02d}_{dim:03d}"] = float(text[pos, dim])
        for pos in range(self.top_k):
            for dim in range(self.image_dim):
                row[f"skapp_graph_image_{pos:02d}_{dim:04d}"] = float(image[pos, dim])
        return row

    def _pair_features(
        self,
        row: pd.Series,
        candidate_idx: int,
        sparse_score: float,
        dense_score: float,
        rrf_score: float,
    ) -> List[float]:
        candidate = self.train_df.iloc[candidate_idx]
        query_genres = set(_parse_literal_list(row.get("genres")))
        candidate_genres = set(_parse_literal_list(candidate.get("genres")))
        query_studios = set(_parse_studios(row.get("studios")))
        candidate_studios = set(_parse_studios(candidate.get("studios")))
        query_voice = set(_parse_pipe(row.get("voice_actor_names")))
        candidate_voice = set(_parse_pipe(candidate.get("voice_actor_names")))
        query_year = _safe_int(row.get("release_year"), default=0)
        candidate_year = _safe_int(candidate.get("release_year"), default=0)
        query_quarter = _safe_int(row.get("release_quarter"), default=0)
        candidate_quarter = _safe_int(candidate.get("release_quarter"), default=0)
        query_episode = _safe_float(row.get("episodes"), default=0.0)
        candidate_episode = _safe_float(candidate.get("episodes"), default=0.0)
        return [
            float(sparse_score),
            float(dense_score),
            float(rrf_score),
            _jaccard(query_genres, candidate_genres),
            _jaccard(query_studios, candidate_studios),
            _jaccard(query_voice, candidate_voice),
            float(max(query_year - candidate_year, 0)),
            float(abs((query_year * 4 + query_quarter) - (candidate_year * 4 + candidate_quarter))),
            float(_same_value(row.get("format"), candidate.get("format"))),
            float(_same_value(row.get("source"), candidate.get("source"))),
            float(_same_value(row.get("countryOfOrigin"), candidate.get("countryOfOrigin"))),
            math.log1p(max(query_episode, 0.0)),
            math.log1p(max(candidate_episode, 0.0)),
            math.log1p(abs(query_episode - candidate_episode)),
            math.log1p(max(_safe_float(candidate.get("popularity"), 0.0), 0.0)),
            _safe_float(candidate.get("meanScore"), 0.0) / 100.0,
        ]

    def _pair_contribution_label(self, row: pd.Series, candidate_idx: int) -> float:
        candidate = self.train_df.iloc[candidate_idx]
        query_pop = math.log1p(max(_safe_float(row.get("popularity"), 0.0), 0.0))
        candidate_pop = math.log1p(max(_safe_float(candidate.get("popularity"), 0.0), 0.0))
        pop_closeness = math.exp(-abs(query_pop - candidate_pop))
        query_score = _safe_float(row.get("meanScore"), 0.0)
        candidate_score = _safe_float(candidate.get("meanScore"), 0.0)
        score_closeness = math.exp(-abs(query_score - candidate_score) / 10.0)
        return float(0.5 * pop_closeness + 0.5 * score_closeness)

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

    def _load_image_embeddings(self) -> dict:
        emb_dir = Path(self.data_cfg["image_emb_dir"])
        emb_cfg = self.config["features"]["image_embedding"]
        cache: dict = {"train_matrix": None}
        for split in self.data_cfg.get("splits", ["train", "val", "test"]):
            path = emb_dir / emb_cfg["file_template"].format(split=split)
            if not path.exists():
                continue
            df = pd.read_parquet(path).set_index(self.id_col)
            prefix = emb_cfg.get("prefix")
            if prefix:
                cols = [col for col in df.columns if str(col).startswith(prefix)]
            else:
                cols = [col for col in df.columns if col != self.id_col]
            values = df[cols].astype(np.float32)
            norms = np.linalg.norm(values.values, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = values.values / norms
            id_to_row = {
                int(idx): normalized[row_idx]
                for row_idx, idx in enumerate(values.index.astype(int).tolist())
            }
            cache[split] = id_to_row
            if split == "train":
                order = [int(item) for item in self.train_ids.tolist()]
                train_matrix = np.zeros((len(order), len(cols)), dtype=np.float32)
                for row_idx, anime_id in enumerate(order):
                    vec = id_to_row.get(anime_id)
                    if vec is not None:
                        train_matrix[row_idx] = vec
                cache["train_matrix"] = train_matrix
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


def _safe_float(value, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def _same_value(left, right) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    return str(left).strip() == str(right).strip()


def _softmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float64)
    centered = values - float(np.max(values))
    exp = np.exp(centered)
    denom = float(exp.sum())
    if denom <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return exp / denom


if __name__ == "__main__":
    main()
