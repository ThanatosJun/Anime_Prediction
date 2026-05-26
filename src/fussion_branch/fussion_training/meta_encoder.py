"""
MetaEncoder: fits on fusion_meta_clean_train + rag_features_train,
transforms any split into a fixed-length float32 numpy array.

Feature layout (in order):
  [standardize]        6 dims  release_year, episodes, duration, startDate_day,
                                prequel_count, prequel_meanScore_mean
  [log1p+standardize]  1 dim   prequel_popularity_mean
  [cyclical sin+cos]   4 dims  release_quarter(period=4), startDate_month(period=12)
  [one-hot format]     7 dims
  [one-hot source]     7 dims
  [one-hot country]    4 dims
  [binary]             3 dims  isAdult, is_sequel, has_sequel
  [genres multi-hot]  19 dims
  [studio TE]          2 dims  standardized mean_pop, mean_score of this anime's studios
  [is_new_studio]      1 dim   1 if all studios are OOV (not in training set), else 0
  [va TE]              2 dims  standardized mean_pop, mean_score of this anime's voice actors
  [rag numerical]      4 dims  rag_popularity, rag_score, rag_release_year, rag_episodes
  [rag_found]          1 dim
  [studio_match]       1 dim   any studio overlap between meta and RAG result (binary)
  [genre_overlap]      1 dim   Jaccard similarity of meta genres and RAG genres
  [format_match]       1 dim   meta format == RAG format (binary)
  [rag_studio TE]      2 dims  standardized mean_pop, mean_score of RAG result's studios

Total: 66 dims
"""
import ast
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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


def _te_lookup(
    names: List[str],
    te_table: Dict[str, Dict[str, float]],
    fallback_pop: float,
    fallback_score: float,
) -> Tuple[float, float]:
    """Look up target encoding for a list of names → (mean_pop, mean_score) raw."""
    if not names:
        return fallback_pop, fallback_score
    pops   = [te_table[n]["pop"]   if n in te_table else fallback_pop   for n in names]
    scores = [te_table[n]["score"] if n in te_table else fallback_score for n in names]
    return float(np.mean(pops)), float(np.mean(scores))


def _standardize(raw_pop: float, raw_score: float, te_stats: Dict[str, float]) -> Tuple[float, float]:
    pop_z   = (raw_pop   - te_stats["pop_mean"])   / te_stats["pop_std"]
    score_z = (raw_score - te_stats["score_mean"]) / te_stats["score_std"]
    return float(pop_z), float(score_z)


class MetaEncoder:
    def __init__(self):
        self.std_medians: Dict[str, float] = {}
        self.std_means:   Dict[str, float] = {}
        self.std_stds:    Dict[str, float] = {}
        self.cyc_medians: Dict[str, float] = {}
        self.cat_vocabs:  Dict[str, List[str]] = {}
        self.genre_vocab: List[str] = []

        # target encoding tables (name → {pop, score} raw train means)
        self.studio_te:   Dict[str, Dict[str, float]] = {}
        self.va_te:       Dict[str, Dict[str, float]] = {}
        self.te_fallback: Dict[str, float] = {}   # {pop, score} — train overall mean
        self.te_stats:    Dict[str, float] = {}   # standardization params

        # rag numerical
        self.rag_medians: Dict[str, float] = {}
        self.rag_means:   Dict[str, float] = {}
        self.rag_stds:    Dict[str, float] = {}

        self.feature_dim: int = 0

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, meta_df: pd.DataFrame, rag_df: pd.DataFrame) -> "MetaEncoder":
        # numerical standardization
        for col in STANDARDIZE_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.std_medians[col] = float(s.median())
            s = s.fillna(self.std_medians[col])
            self.std_means[col] = float(s.mean())
            std = float(s.std())
            self.std_stds[col] = std if std > 0 else 1.0

        for col in LOG1P_STANDARDIZE_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.std_medians[col] = float(s.median())
            s = np.log1p(s.fillna(self.std_medians[col]).values)
            self.std_means[col] = float(s.mean())
            std = float(s.std())
            self.std_stds[col] = std if std > 0 else 1.0

        for col in CYCLICAL_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.cyc_medians[col] = float(s.median())

        # categorical vocabs (for one-hot)
        for col in CATEGORICAL_COLS:
            self.cat_vocabs[col] = sorted(meta_df[col].dropna().unique().tolist())

        # genres multi-hot vocab
        genres: set = set()
        for v in meta_df["genres"]:
            genres.update(_parse_genres(v))
        self.genre_vocab = sorted(genres)

        # target values for TE — raw popularity (z-scored internally; log1p would compress signal)
        pop_col   = pd.to_numeric(meta_df["popularity"], errors="coerce")
        score_col = pd.to_numeric(meta_df["meanScore"],  errors="coerce")
        fallback_pop   = float(pop_col.mean())
        fallback_score = float(score_col.mean())
        self.te_fallback = {"pop": fallback_pop, "score": fallback_score}

        # studio target encoding
        studio_pop_acc:   Dict[str, List[float]] = defaultdict(list)
        studio_score_acc: Dict[str, List[float]] = defaultdict(list)
        for studios_val, pop_val, score_val in zip(meta_df["studios"], pop_col, score_col):
            if pd.isna(pop_val) or pd.isna(score_val):
                continue
            for s in _parse_studios_meta(studios_val):
                studio_pop_acc[s].append(float(pop_val))
                studio_score_acc[s].append(float(score_val))
        self.studio_te = {
            s: {"pop": float(np.mean(studio_pop_acc[s])), "score": float(np.mean(studio_score_acc[s]))}
            for s in studio_pop_acc
        }

        # voice actor target encoding
        va_pop_acc:   Dict[str, List[float]] = defaultdict(list)
        va_score_acc: Dict[str, List[float]] = defaultdict(list)
        if "voice_actor_names" in meta_df.columns:
            for va_val, pop_val, score_val in zip(meta_df["voice_actor_names"], pop_col, score_col):
                if pd.isna(pop_val) or pd.isna(score_val):
                    continue
                for va in _parse_voice_actors(va_val):
                    va_pop_acc[va].append(float(pop_val))
                    va_score_acc[va].append(float(score_val))
        self.va_te = {
            va: {"pop": float(np.mean(va_pop_acc[va])), "score": float(np.mean(va_score_acc[va]))}
            for va in va_pop_acc
        }

        # TE standardization stats — computed from per-anime studio TE values on train
        te_pop_vals, te_score_vals = [], []
        for studios_val in meta_df["studios"]:
            studios = _parse_studios_meta(studios_val)
            p, s = _te_lookup(studios, self.studio_te, fallback_pop, fallback_score)
            te_pop_vals.append(p)
            te_score_vals.append(s)
        te_pop_arr   = np.array(te_pop_vals,   dtype=np.float64)
        te_score_arr = np.array(te_score_vals, dtype=np.float64)
        pop_std   = float(te_pop_arr.std())
        score_std = float(te_score_arr.std())
        self.te_stats = {
            "pop_mean":   float(te_pop_arr.mean()),
            "pop_std":    pop_std   if pop_std   > 0 else 1.0,
            "score_mean": float(te_score_arr.mean()),
            "score_std":  score_std if score_std > 0 else 1.0,
        }

        # rag numerical standardization (rag_popularity uses log1p)
        for col in RAG_NUMERICAL_COLS:
            s = pd.to_numeric(rag_df[col], errors="coerce")
            self.rag_medians[col] = float(s.median())
            s = s.fillna(self.rag_medians[col])
            if col == "rag_popularity":
                s = np.log1p(s)
            self.rag_means[col] = float(s.mean())
            std = float(s.std())
            self.rag_stds[col] = std if std > 0 else 1.0

        self._update_dim()
        return self

    def _update_dim(self):
        self.feature_dim = (
            len(STANDARDIZE_COLS)                           #  6
            + len(LOG1P_STANDARDIZE_COLS)                   #  1
            + len(CYCLICAL_COLS) * 2                        #  4
            + sum(len(v) for v in self.cat_vocabs.values()) # 18  (7+7+4)
            + len(BOOL_COLS)                                #  3
            + len(self.genre_vocab)                         # 19
            + 2                                             #  studio TE
            + 1                                             #  is_new_studio
            + 2                                             #  va TE
            + len(RAG_NUMERICAL_COLS)                       #  4
            + 1                                             #  rag_found
            + 1                                             #  studio_match
            + 1                                             #  genre_overlap
            + 1                                             #  format_match
            + 2                                             #  rag_studio TE
        )

    @property
    def feature_names_(self) -> List[str]:
        """Ordered list of feature names matching each column in transform() output."""
        names: List[str] = []
        for col in STANDARDIZE_COLS:
            names.append(col)
        for col in LOG1P_STANDARDIZE_COLS:
            names.append(f"{col}_log1p")
        for col in CYCLICAL_COLS:
            names.append(f"{col}_sin")
            names.append(f"{col}_cos")
        for col in CATEGORICAL_COLS:
            for val in self.cat_vocabs.get(col, []):
                names.append(f"{col}_{val}")
        for col in BOOL_COLS:
            names.append(col)
        for g in self.genre_vocab:
            names.append(f"genre_{g}")
        names += ["studio_te_pop", "studio_te_score", "is_new_studio"]
        names += ["va_te_pop", "va_te_score"]
        for col in RAG_NUMERICAL_COLS:
            names.append(f"rag_{col}" if not col.startswith("rag_") else col)
        names.append("rag_found")
        names += ["studio_match", "genre_overlap", "format_match"]
        names += ["rag_studio_te_pop", "rag_studio_te_score"]
        return names

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, meta_df: pd.DataFrame, rag_df: pd.DataFrame) -> np.ndarray:
        N = len(meta_df)
        parts = []
        fallback_pop   = self.te_fallback["pop"]
        fallback_score = self.te_fallback["score"]

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

        # studio target encoding (2 dims) + is_new_studio (1 dim)
        studio_te_mat  = np.zeros((N, 2), dtype=np.float32)
        is_new_studio  = np.zeros((N, 1), dtype=np.float32)
        for i, val in enumerate(meta_df["studios"]):
            studios = _parse_studios_meta(val)
            raw_pop, raw_score = _te_lookup(studios, self.studio_te, fallback_pop, fallback_score)
            std_pop, std_score = _standardize(raw_pop, raw_score, self.te_stats)
            studio_te_mat[i, 0] = std_pop
            studio_te_mat[i, 1] = std_score
            # 1 if ALL studios are OOV (not seen in training), else 0
            if studios and not any(s in self.studio_te for s in studios):
                is_new_studio[i, 0] = 1.0
        parts.append(studio_te_mat)
        parts.append(is_new_studio)

        # voice actor target encoding (2 dims)
        va_te_mat = np.zeros((N, 2), dtype=np.float32)
        va_col = meta_df["voice_actor_names"] if "voice_actor_names" in meta_df.columns else pd.Series([None] * N)
        for i, val in enumerate(va_col):
            vas = _parse_voice_actors(val)
            raw_pop, raw_score = _te_lookup(vas, self.va_te, fallback_pop, fallback_score)
            std_pop, std_score = _standardize(raw_pop, raw_score, self.te_stats)
            va_te_mat[i, 0] = std_pop
            va_te_mat[i, 1] = std_score
        parts.append(va_te_mat)

        # rag numerical (rag_popularity uses log1p, consistent with fit)
        rag_num_mat = np.zeros((N, len(RAG_NUMERICAL_COLS)), dtype=np.float32)
        for j, col in enumerate(RAG_NUMERICAL_COLS):
            s = pd.to_numeric(rag_df[col], errors="coerce").fillna(self.rag_medians[col])
            if col == "rag_popularity":
                s = np.log1p(s)
            rag_num_mat[:, j] = (s.values - self.rag_means[col]) / self.rag_stds[col]
        parts.append(rag_num_mat)

        # rag_found
        found_vec = rag_df["rag_found"].fillna(False).astype(float).values.reshape(N, 1).astype(np.float32)
        parts.append(found_vec)

        # overlap scalars (3 dims: studio_match, genre_overlap, format_match)
        overlap_mat = np.zeros((N, 3), dtype=np.float32)
        meta_studios_list  = [set(_parse_studios_meta(v))  for v in meta_df["studios"]]
        meta_genres_list   = [set(_parse_genres(v))        for v in meta_df["genres"]]
        meta_formats_list  = [str(v) if pd.notna(v) else "" for v in meta_df["format"]]
        rag_studios_list   = [set(_parse_studios_rag(v))   for v in rag_df["rag_studios"]]
        rag_genres_list    = [set(_parse_rag_genres(v))    for v in rag_df["rag_genres"]]
        rag_formats_list   = [str(v) if pd.notna(v) else "" for v in rag_df["rag_format"]]

        for i in range(N):
            # studio_match: binary — any shared studio
            overlap_mat[i, 0] = 1.0 if meta_studios_list[i] & rag_studios_list[i] else 0.0

            # genre_overlap: Jaccard
            union = meta_genres_list[i] | rag_genres_list[i]
            overlap_mat[i, 1] = (
                len(meta_genres_list[i] & rag_genres_list[i]) / len(union)
                if union else 0.0
            )

            # format_match: binary
            mf, rf = meta_formats_list[i], rag_formats_list[i]
            overlap_mat[i, 2] = 1.0 if mf and mf == rf else 0.0
        parts.append(overlap_mat)

        # rag_studio target encoding (2 dims)
        rag_studio_te_mat = np.zeros((N, 2), dtype=np.float32)
        for i, val in enumerate(rag_df["rag_studios"]):
            rag_studios = _parse_studios_rag(val)
            raw_pop, raw_score = _te_lookup(rag_studios, self.studio_te, fallback_pop, fallback_score)
            std_pop, std_score = _standardize(raw_pop, raw_score, self.te_stats)
            rag_studio_te_mat[i, 0] = std_pop
            rag_studio_te_mat[i, 1] = std_score
        parts.append(rag_studio_te_mat)

        return np.concatenate(parts, axis=1)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "std_medians": self.std_medians,
            "std_means":   self.std_means,
            "std_stds":    self.std_stds,
            "cyc_medians": self.cyc_medians,
            "cat_vocabs":  self.cat_vocabs,
            "genre_vocab": self.genre_vocab,
            "studio_te":   self.studio_te,
            "va_te":       self.va_te,
            "te_fallback": self.te_fallback,
            "te_stats":    self.te_stats,
            "rag_medians": self.rag_medians,
            "rag_means":   self.rag_means,
            "rag_stds":    self.rag_stds,
            "feature_dim": self.feature_dim,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetaEncoder":
        with open(path) as f:
            state = json.load(f)
        enc = cls()
        for k, v in state.items():
            setattr(enc, k, v)
        return enc
