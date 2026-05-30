"""
MetaEncoder v2：純 metadata → 56-dim float32 array

v1（66-dim）與 v2（56-dim）差異：
  移除 RAG scalar features（10 dims）：
    rag_popularity, rag_score, rag_release_year, rag_episodes,
    rag_found, studio_match, genre_overlap, format_match, rag_studio TE ×2
  原因：RAG 資訊改由 Cross Attention 輸入，不再進 MetaEncoder

Feature layout（56 dims）：
  [standardize]        6  release_year, episodes, duration, startDate_day,
                          prequel_count, prequel_meanScore_mean
  [log1p+standardize]  1  prequel_popularity_mean
  [cyclical sin+cos]   4  release_quarter(period=4), startDate_month(period=12)
  [one-hot format]     7
  [one-hot source]     7
  [one-hot country]    4
  [binary]             3  isAdult, is_sequel, has_sequel
  [genres multi-hot]  19
  [studio TE]          2  standardized mean_pop, mean_score
  [is_new_studio]      1
  [va TE]              2  standardized mean_pop, mean_score
  ─────────────────────────────────────────────
  Total               56
"""

import ast
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

STANDARDIZE_COLS     = ["release_year", "episodes", "duration", "startDate_day",
                         "prequel_count", "prequel_meanScore_mean"]
LOG1P_STANDARDIZE_COLS = ["prequel_popularity_mean"]
CYCLICAL_COLS        = {"release_quarter": 4, "startDate_month": 12}
CATEGORICAL_COLS     = ["format", "source", "countryOfOrigin"]
BOOL_COLS            = ["isAdult", "is_sequel", "has_sequel"]


def _parse_genres(val) -> List[str]:
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def _parse_studios(val) -> List[str]:
    if pd.isna(val):
        return []
    try:
        return [item["node"]["name"] for item in json.loads(str(val)) if "node" in item]
    except Exception:
        return []


def _parse_voice_actors(val) -> List[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [v.strip() for v in str(val).split("|") if v.strip()]


def _te_lookup(names, te_table, fallback_pop, fallback_score):
    if not names:
        return fallback_pop, fallback_score
    pops   = [te_table[n]["pop"]   if n in te_table else fallback_pop   for n in names]
    scores = [te_table[n]["score"] if n in te_table else fallback_score for n in names]
    return float(np.mean(pops)), float(np.mean(scores))


def _standardize_te(raw_pop, raw_score, te_stats):
    pop_z   = (raw_pop   - te_stats["pop_center"])   / te_stats["pop_scale"]
    score_z = (raw_score - te_stats["score_center"]) / te_stats["score_scale"]
    return float(pop_z), float(score_z)


class MetaEncoder:
    def __init__(self):
        self.std_medians: Dict[str, float] = {}
        self.std_centers: Dict[str, float] = {}
        self.std_scales:  Dict[str, float] = {}
        self.cyc_medians: Dict[str, float] = {}
        self.cat_vocabs:  Dict[str, List[str]] = {}
        self.genre_vocab: List[str] = []
        self.studio_te:   Dict[str, Dict[str, float]] = {}
        self.va_te:       Dict[str, Dict[str, float]] = {}
        self.te_fallback: Dict[str, float] = {}
        self.te_stats:    Dict[str, float] = {}
        self.feature_dim: int = 0

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, meta_df: pd.DataFrame) -> "MetaEncoder":
        # robust standardization: (x − median) / IQR
        for col in STANDARDIZE_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.std_medians[col] = float(s.median())
            s = s.fillna(self.std_medians[col])
            self.std_centers[col] = float(s.median())
            q75, q25 = float(np.percentile(s, 75)), float(np.percentile(s, 25))
            self.std_scales[col] = (q75 - q25) if (q75 - q25) > 0 else 1.0

        for col in LOG1P_STANDARDIZE_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.std_medians[col] = float(s.median())
            s = np.log1p(s.fillna(self.std_medians[col]).values)
            self.std_centers[col] = float(np.median(s))
            q75, q25 = float(np.percentile(s, 75)), float(np.percentile(s, 25))
            self.std_scales[col] = (q75 - q25) if (q75 - q25) > 0 else 1.0

        for col in CYCLICAL_COLS:
            s = pd.to_numeric(meta_df[col], errors="coerce")
            self.cyc_medians[col] = float(s.median())

        for col in CATEGORICAL_COLS:
            self.cat_vocabs[col] = sorted(meta_df[col].dropna().unique().tolist())

        genres: set = set()
        for v in meta_df["genres"]:
            genres.update(_parse_genres(v))
        self.genre_vocab = sorted(genres)

        pop_col   = pd.to_numeric(meta_df["popularity"], errors="coerce")
        score_col = pd.to_numeric(meta_df["meanScore"],  errors="coerce")
        fallback_pop   = float(pop_col.median())
        fallback_score = float(score_col.median())
        self.te_fallback = {"pop": fallback_pop, "score": fallback_score}

        # studio target encoding
        studio_pop_acc:   Dict[str, List[float]] = defaultdict(list)
        studio_score_acc: Dict[str, List[float]] = defaultdict(list)
        for studios_val, pop_val, score_val in zip(meta_df["studios"], pop_col, score_col):
            if pd.isna(pop_val) or pd.isna(score_val):
                continue
            for s in _parse_studios(studios_val):
                studio_pop_acc[s].append(float(pop_val))
                studio_score_acc[s].append(float(score_val))
        self.studio_te = {
            s: {"pop": float(np.mean(studio_pop_acc[s])),
                "score": float(np.mean(studio_score_acc[s]))}
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
            va: {"pop": float(np.mean(va_pop_acc[va])),
                 "score": float(np.mean(va_score_acc[va]))}
            for va in va_pop_acc
        }

        # TE robust standardization stats
        te_pop_vals, te_score_vals = [], []
        for studios_val in meta_df["studios"]:
            p, s = _te_lookup(_parse_studios(studios_val), self.studio_te,
                              fallback_pop, fallback_score)
            te_pop_vals.append(p)
            te_score_vals.append(s)
        te_pop_arr   = np.array(te_pop_vals,   dtype=np.float64)
        te_score_arr = np.array(te_score_vals, dtype=np.float64)
        pop_q75,   pop_q25   = float(np.percentile(te_pop_arr,   75)), float(np.percentile(te_pop_arr,   25))
        score_q75, score_q25 = float(np.percentile(te_score_arr, 75)), float(np.percentile(te_score_arr, 25))
        self.te_stats = {
            "pop_center":   float(np.median(te_pop_arr)),
            "pop_scale":    (pop_q75 - pop_q25)     if (pop_q75 - pop_q25)     > 0 else 1.0,
            "score_center": float(np.median(te_score_arr)),
            "score_scale":  (score_q75 - score_q25) if (score_q75 - score_q25) > 0 else 1.0,
        }

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
        )                                                   # = 56

    @property
    def feature_names_(self) -> List[str]:
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
        return names

    # ── transform ─────────────────────────────────────────────────────────────

    def transform(self, meta_df: pd.DataFrame) -> np.ndarray:
        N = len(meta_df)
        parts = []
        fallback_pop   = self.te_fallback["pop"]
        fallback_score = self.te_fallback["score"]

        # robust standardize
        std_mat = np.zeros((N, len(STANDARDIZE_COLS)), dtype=np.float32)
        for j, col in enumerate(STANDARDIZE_COLS):
            s = pd.to_numeric(meta_df[col], errors="coerce").fillna(self.std_medians[col])
            std_mat[:, j] = (s.values - self.std_centers[col]) / self.std_scales[col]
        parts.append(std_mat)

        # log1p + robust standardize
        log_mat = np.zeros((N, len(LOG1P_STANDARDIZE_COLS)), dtype=np.float32)
        for j, col in enumerate(LOG1P_STANDARDIZE_COLS):
            s = pd.to_numeric(meta_df[col], errors="coerce").fillna(self.std_medians[col])
            s = np.log1p(s.values)
            log_mat[:, j] = (s - self.std_centers[col]) / self.std_scales[col]
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
        g_idx     = {g: i for i, g in enumerate(self.genre_vocab)}
        genre_mat = np.zeros((N, len(self.genre_vocab)), dtype=np.float32)
        for i, val in enumerate(meta_df["genres"]):
            for g in _parse_genres(val):
                if g in g_idx:
                    genre_mat[i, g_idx[g]] = 1.0
        parts.append(genre_mat)

        # studio TE (2) + is_new_studio (1)
        studio_te_mat = np.zeros((N, 2), dtype=np.float32)
        is_new_studio = np.zeros((N, 1), dtype=np.float32)
        for i, val in enumerate(meta_df["studios"]):
            studios = _parse_studios(val)
            raw_pop, raw_score = _te_lookup(studios, self.studio_te, fallback_pop, fallback_score)
            studio_te_mat[i, 0], studio_te_mat[i, 1] = _standardize_te(raw_pop, raw_score, self.te_stats)
            if studios and not any(s in self.studio_te for s in studios):
                is_new_studio[i, 0] = 1.0
        parts.append(studio_te_mat)
        parts.append(is_new_studio)

        # va TE (2)
        va_te_mat = np.zeros((N, 2), dtype=np.float32)
        va_col = meta_df["voice_actor_names"] if "voice_actor_names" in meta_df.columns else pd.Series([None] * N)
        for i, val in enumerate(va_col):
            vas = _parse_voice_actors(val)
            raw_pop, raw_score = _te_lookup(vas, self.va_te, fallback_pop, fallback_score)
            va_te_mat[i, 0], va_te_mat[i, 1] = _standardize_te(raw_pop, raw_score, self.te_stats)
        parts.append(va_te_mat)

        return np.concatenate(parts, axis=1)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "std_medians": self.std_medians,
            "std_centers": self.std_centers,
            "std_scales":  self.std_scales,
            "cyc_medians": self.cyc_medians,
            "cat_vocabs":  self.cat_vocabs,
            "genre_vocab": self.genre_vocab,
            "studio_te":   self.studio_te,
            "va_te":       self.va_te,
            "te_fallback": self.te_fallback,
            "te_stats":    self.te_stats,
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
