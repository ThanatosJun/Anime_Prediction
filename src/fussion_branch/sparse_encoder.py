import ast
import json
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


class SparseEncoder:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.dim: int = 0

    def fit(self, df: pd.DataFrame) -> "SparseEncoder":
        tokens: set = set()
        for g in df["genres"]:
            for name in parse_genres(g):
                tokens.add(f"genre:{name}")
        for s in df["studios"]:
            for name in parse_studios(s):
                tokens.add(f"studio:{name}")
        self.vocab = {tok: i for i, tok in enumerate(sorted(tokens))}
        self.dim = len(self.vocab)
        return self

    def encode(self, genres: List[str], studios: List[str]) -> Tuple[List[int], List[float]]:
        seen: set = set()
        indices: List[int] = []
        values: List[float] = []
        for tok in [f"genre:{g}" for g in genres] + [f"studio:{s}" for s in studios]:
            if tok in self.vocab and tok not in seen:
                indices.append(self.vocab[tok])
                values.append(1.0)
                seen.add(tok)
        return indices, values

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str) -> "SparseEncoder":
        with open(path) as f:
            data = json.load(f)
        enc = cls()
        enc.vocab = data["vocab"]
        enc.dim = data["dim"]
        return enc
