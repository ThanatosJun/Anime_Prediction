"""
MetaEncoder: fits on fusion_meta_clean_train + rag_features_train,
transforms any split into a fixed-length float32 numpy array.

Feature layout (in order):
  [standardize]        6 dims  release_year, episodes, duration, startDate_day,
                                prequel_count, prequel_meanScore_mean
  [log1p+standardize]  1 dim   prequel_popularity_mean
  [cyclical sin+cos]   4 dims  release_quarter(period=4), startDate_month(period=12)
  [one-hot format]     7 dims  fit from training vocab
  [one-hot source]     7 dims  fit from training vocab
  [one-hot country]    4 dims  fit from training vocab
  [binary]             3 dims  isAdult, is_sequel, has_sequel
  [genres multi-hot]  19 dims
  [studios multi-hot]       50 dims  top-K by frequency
  [voice_actors multi-hot]  50 dims  top-K by frequency (NaN → all zeros)
  [rag numerical]            4 dims  rag_popularity, rag_score, rag_release_year, rag_episodes (standardized)
  [rag_found]          1 dim
  [rag studios]       50 dims  same studio vocab as meta
  [rag genres]        19 dims  same genre vocab as meta
  [rag format]         7 dims  same format vocab as meta
"""
import ast
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

STANDARDIZE_COLS = [
    "release_year", "episodes", "duration", "startDate_day",
    "prequel_count", "prequel_meanScore_mean",
]
LOG1P_STANDARDIZE_COLS = ["prequel_popularity_mean"]
CYCLICAL_COLS = {"release_quarter": 4, "startDate_month": 12}
CATEGORICAL_COLS = ["format", "source", "countryOfOrigin"]
BOOL_COLS = ["isAdult", "is_sequel", "has_sequel"]
RAG_NUMERICAL_COLS = ["rag_popularity", "rag_score", "rag_release_year", "rag_episodes"]


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


def _parse_voice_actors(val) -> List[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [v.strip() for v in str(val).split("|") if v.strip()]


def _parse_rag_genres(val) -> List[str]:
    if pd.isna(val) or val == "[]":
        return []
    try:
        return json.loads(str(val))
    except Exception:
        return []


class MetaEncoder:
    def __init__(self, top_studios: int = 50, top_voice_actors: int = 50):
        self.top_studios       = top_studios
        self.top_voice_actors  = top_voice_actors
        self.std_medians:  Dict[str, float] = {}
        self.std_means:    Dict[str, float] = {}
        self.std_stds:     Dict[str, float] = {}
        self.cyc_medians:  Dict[str, float] = {}
        self.cat_vocabs:   Dict[str, List[str]] = {}
        self.genre_vocab:        List[str] = []
        self.studio_vocab:       List[str] = []
        self.voice_actor_vocab:  List[str] = []
        self.rag_medians:  Dict[str, float] = {}
        self.rag_means:    Dict[str, float] = {}
        self.rag_stds:     Dict[str, float] = {}
        self.feature_dim:  int = 0

    def fit(self, meta_df: pd.DataFrame, rag_df: pd.DataFrame) -> "MetaEncoder":
        # standardize cols: median imputation + mean/std
        for col in STANDARDIZE_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.std_medians[col] = float(s.median())
            s = s.fillna(self.std_medians[col])
            self.std_means[col] = float(s.mean())
            std = float(s.std())
            self.std_stds[col] = std if std > 0 else 1.0

        # log1p + standardize cols
        for col in LOG1P_STANDARDIZE_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.std_medians[col] = float(s.median())
            s = np.log1p(s.fillna(self.std_medians[col]).values)
            self.std_means[col] = float(s.mean())
            std = float(s.std())
            self.std_stds[col] = std if std > 0 else 1.0

        # cyclical cols: store median for NaN imputation
        for col in CYCLICAL_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.cyc_medians[col] = float(s.median())

        # categorical one-hot: vocab from training (unknown → all zeros)
        for col in CATEGORICAL_COLS:
            self.cat_vocabs[col] = sorted(meta_df[col].dropna().unique().tolist())

        # genres multi-hot
        genres: set = set()
        for v in meta_df["genres"]:
            genres.update(_parse_genres(v))
        self.genre_vocab = sorted(genres)

        # studios top-K by frequency
        from collections import Counter
        cnt: Counter = Counter()
        for v in meta_df["studios"]:
            for s in _parse_studios_meta(v):
                cnt[s] += 1
        self.studio_vocab = [s for s, _ in cnt.most_common(self.top_studios)]

        # voice_actors top-K by frequency
        va_cnt: Counter = Counter()
        for v in meta_df["voice_actor_names"]:
            for va in _parse_voice_actors(v):
                va_cnt[va] += 1
        self.voice_actor_vocab = [va for va, _ in va_cnt.most_common(self.top_voice_actors)]

        # rag numerical: median imputation + standardize
        for col in RAG_NUMERICAL_COLS:
            s = pd.to_numeric(rag_df[col], errors="coerce")
            self.rag_medians[col] = float(s.median())
            s = s.fillna(self.rag_medians[col])
            self.rag_means[col] = float(s.mean())
            std = float(s.std())
            self.rag_stds[col] = std if std > 0 else 1.0

        self._update_dim()
        return self

    def _update_dim(self):
        self.feature_dim = (
            len(STANDARDIZE_COLS)
            + len(LOG1P_STANDARDIZE_COLS)
            + len(CYCLICAL_COLS) * 2          # sin + cos per col
            + sum(len(v) for v in self.cat_vocabs.values())
            + len(BOOL_COLS)
            + len(self.genre_vocab)
            + len(self.studio_vocab)
            + len(self.voice_actor_vocab)     # voice_actors multi-hot
            + len(RAG_NUMERICAL_COLS)         # rag_popularity, rag_score, rag_release_year, rag_episodes
            + 1                               # rag_found
            + len(self.studio_vocab)          # rag_studios
            + len(self.genre_vocab)           # rag_genres
            + len(self.cat_vocabs.get("format", []))  # rag_format
        )

    def transform(self, meta_df: pd.DataFrame, rag_df: pd.DataFrame) -> np.ndarray:
        N = len(meta_df)
        parts = []

        # standardize
        std_mat = np.zeros((N, len(STANDARDIZE_COLS)), dtype=np.float32)
        for j, col in enumerate(STANDARDIZE_COLS):
            s = pd.to_numeric(meta_df[col], errors="coerce").fillna(self.std_medians[col])
            std_mat[:, j] = (s.values - self.std_means[col]) / self.std_stds[col]
        parts.append(std_mat)

        # log1p + standardize
        log_mat = np.zeros((N, len(LOG1P_STANDARDIZE_COLS)), dtype=np.float32)
        for j, col in enumerate(LOG1P_STANDARDIZE_COLS):
            s = pd.to_numeric(meta_df[col], errors="coerce").fillna(self.std_medians[col])
            s = np.log1p(s.values)
            log_mat[:, j] = (s - self.std_means[col]) / self.std_stds[col]
        parts.append(log_mat)

        # cyclical sin + cos
        cyc_mat = np.zeros((N, len(CYCLICAL_COLS) * 2), dtype=np.float32)
        for j, (col, period) in enumerate(CYCLICAL_COLS.items()):
            s = pd.to_numeric(meta_df[col], errors="coerce").fillna(self.cyc_medians[col]).values
            cyc_mat[:, j * 2]     = np.sin(2 * math.pi * s / period).astype(np.float32)
            cyc_mat[:, j * 2 + 1] = np.cos(2 * math.pi * s / period).astype(np.float32)
        parts.append(cyc_mat)

        # categorical one-hot
        for col in CATEGORICAL_COLS:
            vocab = self.cat_vocabs[col]
            mat = np.zeros((N, len(vocab)), dtype=np.float32)
            for i, val in enumerate(meta_df[col]):
                if val in vocab:
                    mat[i, vocab.index(val)] = 1.0
            parts.append(mat)

        # binary
        bool_mat = np.zeros((N, len(BOOL_COLS)), dtype=np.float32)
        for j, col in enumerate(BOOL_COLS):
            bool_mat[:, j] = meta_df[col].fillna(False).astype(float).values
        parts.append(bool_mat)

        # genres multi-hot
        g_idx = {g: i for i, g in enumerate(self.genre_vocab)}
        genre_mat = np.zeros((N, len(self.genre_vocab)), dtype=np.float32)
        for i, val in enumerate(meta_df["genres"]):
            for g in _parse_genres(val):
                if g in g_idx:
                    genre_mat[i, g_idx[g]] = 1.0
        parts.append(genre_mat)

        # studios multi-hot
        s_idx = {s: i for i, s in enumerate(self.studio_vocab)}
        studio_meta_mat = np.zeros((N, len(self.studio_vocab)), dtype=np.float32)
        for i, val in enumerate(meta_df["studios"]):
            for s in _parse_studios_meta(val):
                if s in s_idx:
                    studio_meta_mat[i, s_idx[s]] = 1.0
        parts.append(studio_meta_mat)

        # voice_actors multi-hot (NaN → all zeros)
        va_idx = {va: i for i, va in enumerate(self.voice_actor_vocab)}
        va_mat = np.zeros((N, len(self.voice_actor_vocab)), dtype=np.float32)
        for i, val in enumerate(meta_df["voice_actor_names"]):
            for va in _parse_voice_actors(val):
                if va in va_idx:
                    va_mat[i, va_idx[va]] = 1.0
        parts.append(va_mat)

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

        # rag genres multi-hot (same vocab as meta genres)
        rag_genre_mat = np.zeros((N, len(self.genre_vocab)), dtype=np.float32)
        for i, val in enumerate(rag_df["rag_genres"]):
            for g in _parse_rag_genres(val):
                if g in g_idx:
                    rag_genre_mat[i, g_idx[g]] = 1.0
        parts.append(rag_genre_mat)

        # rag format one-hot (same vocab as meta format)
        format_vocab = self.cat_vocabs.get("format", [])
        rag_format_mat = np.zeros((N, len(format_vocab)), dtype=np.float32)
        for i, val in enumerate(rag_df["rag_format"]):
            if pd.notna(val) and val in format_vocab:
                rag_format_mat[i, format_vocab.index(val)] = 1.0
        parts.append(rag_format_mat)

        return np.concatenate(parts, axis=1)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "top_studios":        self.top_studios,
            "top_voice_actors":   self.top_voice_actors,
            "std_medians":        self.std_medians,
            "std_means":          self.std_means,
            "std_stds":           self.std_stds,
            "cyc_medians":        self.cyc_medians,
            "cat_vocabs":         self.cat_vocabs,
            "genre_vocab":        self.genre_vocab,
            "studio_vocab":       self.studio_vocab,
            "voice_actor_vocab":  self.voice_actor_vocab,
            "rag_medians":        self.rag_medians,
            "rag_means":          self.rag_means,
            "rag_stds":           self.rag_stds,
            "feature_dim":        self.feature_dim,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetaEncoder":
        with open(path) as f:
            state = json.load(f)
        enc = cls(top_studios=state["top_studios"], top_voice_actors=state.get("top_voice_actors", 50))
        for k, v in state.items():
            setattr(enc, k, v)
        return enc
