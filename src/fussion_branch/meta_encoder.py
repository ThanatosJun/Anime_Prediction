"""
MetaEncoder: fits on fusion_meta_train + rag_features_train,
transforms any split into a fixed-length float32 numpy array.

Feature layout (in order):
  [meta_numerical]       11 dims  (standardized, NaN → train median)
  [meta_format]           7 dims  (one-hot)
  [meta_source]           6 dims  (one-hot)
  [meta_countryOfOrigin]  4 dims  (one-hot)
  [meta_season]           4 dims  (one-hot)
  [meta_bool]             3 dims  (isAdult, is_sequel, has_sequel)
  [meta_genres]          19 dims  (multi-hot)
  [meta_studios]         50 dims  (top-K multi-hot)
  [rag_numerical]         3 dims  (rag_popularity, rag_score, rag_release_year — standardized)
  [rag_found]             1 dim
  [rag_studios]          50 dims  (same studio vocab as meta_studios)
"""
import ast
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

NUMERICAL_COLS = [
    "release_year", "release_quarter", "seasonYear",
    "startDate_year", "startDate_month", "startDate_day",
    "episodes", "duration",
    "prequel_count", "prequel_popularity_mean", "prequel_meanScore_mean",
]
CATEGORICAL_COLS = ["format", "source", "countryOfOrigin", "season"]
BOOL_COLS = ["isAdult", "is_sequel", "has_sequel"]
RAG_NUMERICAL_COLS = ["rag_popularity", "rag_score", "rag_release_year"]


def _parse_genres(val) -> List[str]:
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def _parse_studios_meta(val) -> List[str]:
    if pd.isna(val):
        return []
    try:
        return [item["node"]["name"] for item in json.loads(str(val)) if "node" in item]
    except Exception:
        return []


def _parse_studios_rag(val) -> List[str]:
    if pd.isna(val) or val == "[]":
        return []
    try:
        return json.loads(str(val))
    except Exception:
        return []


class MetaEncoder:
    def __init__(self, top_studios: int = 50):
        self.top_studios = top_studios
        # fitted state
        self.num_medians: Dict[str, float] = {}
        self.num_means:   Dict[str, float] = {}
        self.num_stds:    Dict[str, float] = {}
        self.cat_vocabs:  Dict[str, List[str]] = {}
        self.genre_vocab: List[str] = []
        self.studio_vocab: List[str] = []
        self.rag_medians: Dict[str, float] = {}
        self.rag_means:   Dict[str, float] = {}
        self.rag_stds:    Dict[str, float] = {}
        self.feature_dim: int = 0

    def fit(self, meta_df: pd.DataFrame, rag_df: pd.DataFrame) -> "MetaEncoder":
        # numerical: median imputation + standardize
        for col in NUMERICAL_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.num_medians[col] = float(s.median())
            s_filled = s.fillna(self.num_medians[col])
            self.num_means[col] = float(s_filled.mean())
            std = float(s_filled.std())
            self.num_stds[col] = std if std > 0 else 1.0

        # categorical: vocab from training (missing → all zeros)
        for col in CATEGORICAL_COLS:
            self.cat_vocabs[col] = sorted(meta_df[col].dropna().unique().tolist())

        # genre multi-hot vocab
        genres = set()
        for v in meta_df["genres"]:
            genres.update(_parse_genres(v))
        self.genre_vocab = sorted(genres)

        # studio top-K by frequency
        from collections import Counter
        cnt: Counter = Counter()
        for v in meta_df["studios"]:
            for s in _parse_studios_meta(v):
                cnt[s] += 1
        self.studio_vocab = [s for s, _ in cnt.most_common(self.top_studios)]

        # rag numerical: median imputation + standardize
        for col in RAG_NUMERICAL_COLS:
            s = pd.to_numeric(rag_df[col], errors="coerce")
            self.rag_medians[col] = float(s.median())
            s_filled = s.fillna(self.rag_medians[col])
            self.rag_means[col] = float(s_filled.mean())
            std = float(s_filled.std())
            self.rag_stds[col] = std if std > 0 else 1.0

        self._update_dim()
        return self

    def _update_dim(self):
        self.feature_dim = (
            len(NUMERICAL_COLS)
            + sum(len(v) for v in self.cat_vocabs.values())
            + len(BOOL_COLS)
            + len(self.genre_vocab)
            + len(self.studio_vocab)   # meta studios
            + len(RAG_NUMERICAL_COLS)
            + 1                        # rag_found
            + len(self.studio_vocab)   # rag studios (same vocab)
        )

    def transform(self, meta_df: pd.DataFrame, rag_df: pd.DataFrame) -> np.ndarray:
        N = len(meta_df)
        parts = []

        # meta numerical
        num_mat = np.zeros((N, len(NUMERICAL_COLS)), dtype=np.float32)
        for j, col in enumerate(NUMERICAL_COLS):
            s = pd.to_numeric(meta_df[col], errors="coerce").fillna(self.num_medians[col])
            num_mat[:, j] = (s.values - self.num_means[col]) / self.num_stds[col]
        parts.append(num_mat)

        # meta categorical one-hot
        for col in CATEGORICAL_COLS:
            vocab = self.cat_vocabs[col]
            mat = np.zeros((N, len(vocab)), dtype=np.float32)
            for i, val in enumerate(meta_df[col]):
                if val in vocab:
                    mat[i, vocab.index(val)] = 1.0
            parts.append(mat)

        # meta bool
        bool_mat = np.zeros((N, len(BOOL_COLS)), dtype=np.float32)
        for j, col in enumerate(BOOL_COLS):
            bool_mat[:, j] = meta_df[col].fillna(False).astype(float).values
        parts.append(bool_mat)

        # meta genres multi-hot
        g_idx = {g: i for i, g in enumerate(self.genre_vocab)}
        genre_mat = np.zeros((N, len(self.genre_vocab)), dtype=np.float32)
        for i, val in enumerate(meta_df["genres"]):
            for g in _parse_genres(val):
                if g in g_idx:
                    genre_mat[i, g_idx[g]] = 1.0
        parts.append(genre_mat)

        # meta studios multi-hot
        s_idx = {s: i for i, s in enumerate(self.studio_vocab)}
        studio_meta_mat = np.zeros((N, len(self.studio_vocab)), dtype=np.float32)
        for i, val in enumerate(meta_df["studios"]):
            for s in _parse_studios_meta(val):
                if s in s_idx:
                    studio_meta_mat[i, s_idx[s]] = 1.0
        parts.append(studio_meta_mat)

        # rag numerical
        rag_num_mat = np.zeros((N, len(RAG_NUMERICAL_COLS)), dtype=np.float32)
        for j, col in enumerate(RAG_NUMERICAL_COLS):
            s = pd.to_numeric(rag_df[col], errors="coerce").fillna(self.rag_medians[col])
            rag_num_mat[:, j] = (s.values - self.rag_means[col]) / self.rag_stds[col]
        parts.append(rag_num_mat)

        # rag_found
        found_vec = rag_df["rag_found"].fillna(False).astype(float).values.reshape(N, 1).astype(np.float32)
        parts.append(found_vec)

        # rag studios multi-hot (same vocab)
        studio_rag_mat = np.zeros((N, len(self.studio_vocab)), dtype=np.float32)
        for i, val in enumerate(rag_df["rag_studios"]):
            for s in _parse_studios_rag(val):
                if s in s_idx:
                    studio_rag_mat[i, s_idx[s]] = 1.0
        parts.append(studio_rag_mat)

        return np.concatenate(parts, axis=1)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "top_studios":  self.top_studios,
            "num_medians":  self.num_medians,
            "num_means":    self.num_means,
            "num_stds":     self.num_stds,
            "cat_vocabs":   self.cat_vocabs,
            "genre_vocab":  self.genre_vocab,
            "studio_vocab": self.studio_vocab,
            "rag_medians":  self.rag_medians,
            "rag_means":    self.rag_means,
            "rag_stds":     self.rag_stds,
            "feature_dim":  self.feature_dim,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetaEncoder":
        with open(path) as f:
            state = json.load(f)
        enc = cls(top_studios=state["top_studios"])
        for k, v in state.items():
            setattr(enc, k, v)
        return enc
