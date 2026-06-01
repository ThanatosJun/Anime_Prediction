import ast
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_genres(genres_str) -> List[str]:
    if pd.isna(genres_str):
        return []
    try:
        return ast.literal_eval(str(genres_str))
    except Exception:
        return []


def parse_studios(studios_str) -> List[str]:
    if pd.isna(studios_str):
        return []
    try:
        items = json.loads(str(studios_str))
        return [
            item["node"]["name"]
            for item in items
            if "node" in item and "name" in item["node"]
        ]
    except Exception:
        return []


def parse_voice_actors(voice_actor_names) -> List[str]:
    if pd.isna(voice_actor_names):
        return []
    s = str(voice_actor_names).strip()
    if not s:
        return []
    return [name.strip() for name in s.split("|") if name.strip()]


def parse_source(source_str) -> str:
    if pd.isna(source_str):
        return ""
    return str(source_str).strip()


class SparseEncoder:
    """
    混合加權 sparse encoder：
      - genre / studio / source：IDF 加權（稀有 = 重要）
          IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
      - voice_actor：df 正向加權（出演多 = 有名 = 重要）
          w(t) = log(df(t) + 1)
    """

    def __init__(self):
        self.vocab:  Dict[str, int]   = {}
        self.dim:    int              = 0
        self._idf:   Dict[str, float] = {}   # genre / studio / source
        self._va_w:  Dict[str, float] = {}   # voice_actor
        self._N:     int              = 0

    def fit(self, df: pd.DataFrame) -> "SparseEncoder":
        all_doc_tokens: List[List[str]] = []
        for _, row in df.iterrows():
            genres       = parse_genres(row.get("genres", ""))
            studios      = parse_studios(row.get("studios", ""))
            voice_actors = parse_voice_actors(row.get("voice_actor_names", ""))
            source       = parse_source(row.get("source", ""))
            tokens = list(dict.fromkeys(
                [f"genre:{g}"  for g in genres] +
                [f"studio:{s}" for s in studios] +
                [f"voice:{v}"  for v in voice_actors] +
                ([f"source:{source}"] if source else [])
            ))
            all_doc_tokens.append(tokens)

        # vocab
        token_set  = {tok for doc in all_doc_tokens for tok in doc}
        self.vocab = {tok: i for i, tok in enumerate(sorted(token_set))}
        self.dim   = len(self.vocab)

        # document frequency
        self._N = len(all_doc_tokens)
        df_count: Dict[str, int] = {}
        for tokens in all_doc_tokens:
            for tok in tokens:
                df_count[tok] = df_count.get(tok, 0) + 1

        # genre / studio / source → Robertson IDF
        self._idf = {
            tok: math.log((self._N - cnt + 0.5) / (cnt + 0.5) + 1)
            for tok, cnt in df_count.items()
            if not tok.startswith("voice:")
        }

        # voice_actor → log(df + 1)（出演越多越有名）
        self._va_w = {
            tok: math.log(cnt + 1)
            for tok, cnt in df_count.items()
            if tok.startswith("voice:")
        }

        print(f"SparseEncoder fitted: vocab={self.dim}  N={self._N}  "
              f"idf_tokens={len(self._idf)}  va_tokens={len(self._va_w)}")
        return self

    def encode(
        self,
        genres:       List[str],
        studios:      List[str],
        voice_actors: List[str] = [],
        source:       str       = "",
    ) -> Tuple[List[int], List[float]]:
        tokens = list(dict.fromkeys(
            [f"genre:{g}"  for g in genres] +
            [f"studio:{s}" for s in studios] +
            [f"voice:{v}"  for v in voice_actors] +
            ([f"source:{source}"] if source else [])
        ))

        indices: List[int]   = []
        values:  List[float] = []
        for tok in tokens:
            if tok not in self.vocab:
                continue
            if tok.startswith("voice:"):
                weight = self._va_w.get(tok, 0.0)
            else:
                weight = self._idf.get(tok, 0.0)
            if weight > 0:
                indices.append(self.vocab[tok])
                values.append(float(weight))

        return indices, values

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "vocab": self.vocab,
                "dim":   self.dim,
                "idf":   self._idf,
                "va_w":  self._va_w,
                "N":     self._N,
            }, f)

    @classmethod
    def load(cls, path: str) -> "SparseEncoder":
        with open(path) as f:
            data = json.load(f)
        enc        = cls()
        enc.vocab  = data["vocab"]
        enc.dim    = data["dim"]
        enc._idf   = data.get("idf",  {})
        enc._va_w  = data.get("va_w", {})
        enc._N     = data.get("N",    0)
        return enc
