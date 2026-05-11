from __future__ import annotations

import ast
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitFeatures:
    ids: np.ndarray
    x: Optional[np.ndarray]
    y_raw: np.ndarray


class MetadataEncoder:
    def __init__(self, config: dict):
        self.config = config
        self.numeric_stats: Dict[str, Tuple[float, float, float]] = {}
        self.log_stats: Dict[str, Tuple[float, float, float]] = {}
        self.cyclic_medians: Dict[str, float] = {}
        self.categorical_vocabs: Dict[str, List[str]] = {}
        self.multihot_vocabs: Dict[str, List[str]] = {}
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        cfg = self.config
        for col in cfg.get("numeric", []):
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median())
            filled = values.fillna(median).astype(float)
            mean = float(filled.mean())
            std = float(filled.std()) or 1.0
            self.numeric_stats[col] = (median, mean, std)

        for col in cfg.get("log_numeric", []):
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median())
            filled = np.log1p(values.fillna(median).astype(float).values)
            mean = float(filled.mean())
            std = float(filled.std()) or 1.0
            self.log_stats[col] = (median, mean, std)

        for col in cfg.get("cyclic", {}):
            values = pd.to_numeric(df[col], errors="coerce")
            self.cyclic_medians[col] = float(values.median())

        for col in cfg.get("categorical", []):
            vocab = sorted(str(v) for v in df[col].dropna().unique().tolist())
            self.categorical_vocabs[col] = vocab

        for col, spec in cfg.get("multihot", {}).items():
            counter: Counter = Counter()
            for value in df[col]:
                counter.update(_parse_multi_value(value, spec.get("parser", "literal_list")))
            top_k = spec.get("top_k")
            if top_k is None:
                vocab = sorted(counter.keys())
            else:
                vocab = [name for name, _ in counter.most_common(int(top_k))]
            self.multihot_vocabs[col] = vocab

        self.feature_names = self._build_feature_names()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []
        n_rows = len(df)

        for col, (median, mean, std) in self.numeric_stats.items():
            values = pd.to_numeric(df[col], errors="coerce").fillna(median).astype(float)
            parts.append(((values.values - mean) / std).reshape(n_rows, 1).astype(np.float32))

        for col, (median, mean, std) in self.log_stats.items():
            values = pd.to_numeric(df[col], errors="coerce").fillna(median).astype(float)
            transformed = np.log1p(values.values)
            parts.append(((transformed - mean) / std).reshape(n_rows, 1).astype(np.float32))

        for col, period in self.config.get("cyclic", {}).items():
            values = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(self.cyclic_medians[col])
                .astype(float)
                .values
            )
            radians = 2 * math.pi * values / float(period)
            parts.append(np.sin(radians).reshape(n_rows, 1).astype(np.float32))
            parts.append(np.cos(radians).reshape(n_rows, 1).astype(np.float32))

        for col, vocab in self.categorical_vocabs.items():
            index = {value: idx for idx, value in enumerate(vocab)}
            mat = np.zeros((n_rows, len(vocab)), dtype=np.float32)
            for row_idx, value in enumerate(df[col]):
                key = str(value) if pd.notna(value) else None
                if key in index:
                    mat[row_idx, index[key]] = 1.0
            parts.append(mat)

        for col in self.config.get("boolean", []):
            values = df[col].fillna(False).astype(float).values
            parts.append(values.reshape(n_rows, 1).astype(np.float32))

        for col, spec in self.config.get("multihot", {}).items():
            vocab = self.multihot_vocabs[col]
            index = {value: idx for idx, value in enumerate(vocab)}
            mat = np.zeros((n_rows, len(vocab)), dtype=np.float32)
            for row_idx, value in enumerate(df[col]):
                for token in _parse_multi_value(value, spec.get("parser", "literal_list")):
                    if token in index:
                        mat[row_idx, index[token]] = 1.0
            parts.append(mat)

        if not parts:
            return np.empty((n_rows, 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        names.extend(self.numeric_stats.keys())
        names.extend(f"log1p_{col}" for col in self.log_stats)
        for col in self.config.get("cyclic", {}):
            names.extend([f"{col}_sin", f"{col}_cos"])
        for col, vocab in self.categorical_vocabs.items():
            names.extend(f"{col}={value}" for value in vocab)
        names.extend(self.config.get("boolean", []))
        for col, vocab in self.multihot_vocabs.items():
            names.extend(f"{col}={value}" for value in vocab)
        return names


class BaselineFeatureStore:
    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.feature_cfg = config["features"]
        self.id_col = self.data_cfg.get("id_col", "id")
        self.meta: Dict[str, pd.DataFrame] = {}

    def load_metadata(self) -> None:
        meta_dir = Path(self.data_cfg["meta_dir"])
        for split in self.data_cfg.get("splits", ["train", "val", "test"]):
            path = meta_dir / f"fusion_meta_clean_{split}.csv"
            self.meta[split] = pd.read_csv(path)

    def build(
        self,
        feature_set: dict,
        target: str,
    ) -> Tuple[Dict[str, SplitFeatures], List[str], Optional[str]]:
        if not self.meta:
            self.load_metadata()

        split_ids, missing = self._resolve_ids(feature_set)
        if missing:
            return {}, [], missing

        train_df = self._df_for_ids("train", split_ids["train"])
        encoder = None
        feature_names: List[str] = []
        if feature_set.get("metadata", False):
            encoder = MetadataEncoder(self.feature_cfg["metadata"]).fit(train_df)
            feature_names.extend([f"meta:{name}" for name in encoder.feature_names])

        emb_cache = self._load_embedding_cache(feature_set, split_ids)
        for prefix, dims in self._embedding_feature_dims(emb_cache):
            feature_names.extend(f"{prefix}:{i}" for i in range(dims))

        output: Dict[str, SplitFeatures] = {}
        for split, ids in split_ids.items():
            df = self._df_for_ids(split, ids)
            parts: List[np.ndarray] = []
            if encoder is not None:
                parts.append(encoder.transform(df))
            for key in ("text_embedding", "image_embedding"):
                if feature_set.get(key, False):
                    parts.append(emb_cache[key][split].loc[ids].values.astype(np.float32))
            x = None if not parts else np.concatenate(parts, axis=1)
            output[split] = SplitFeatures(
                ids=ids.to_numpy(),
                x=x,
                y_raw=df[target].values.astype(np.float64),
            )
        return output, feature_names, None

    def _resolve_ids(self, feature_set: dict) -> Tuple[Dict[str, pd.Index], Optional[str]]:
        split_ids = {
            split: pd.Index(df[self.id_col].values, name=self.id_col)
            for split, df in self.meta.items()
        }
        for key in ("text_embedding", "image_embedding"):
            if not feature_set.get(key, False):
                continue
            emb_dir_key = "text_emb_dir" if key == "text_embedding" else "image_emb_dir"
            emb_cfg_key = "text_embedding" if key == "text_embedding" else "image_embedding"
            emb_dir = Path(self.data_cfg[emb_dir_key])
            template = self.feature_cfg[emb_cfg_key]["file_template"]
            if not emb_dir.exists():
                return split_ids, f"Missing {key} directory: {emb_dir}"
            for split in split_ids:
                path = emb_dir / template.format(split=split)
                if not path.exists():
                    return split_ids, f"Missing {key} file: {path}"
                emb_ids = pd.Index(pd.read_parquet(path, columns=[self.id_col])[self.id_col].values)
                split_ids[split] = split_ids[split].intersection(emb_ids)
        return split_ids, None

    def _load_embedding_cache(
        self,
        feature_set: dict,
        split_ids: Dict[str, pd.Index],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        for key in ("text_embedding", "image_embedding"):
            if not feature_set.get(key, False):
                continue
            emb_dir_key = "text_emb_dir" if key == "text_embedding" else "image_emb_dir"
            emb_cfg = self.feature_cfg[key]
            emb_dir = Path(self.data_cfg[emb_dir_key])
            cache[key] = {}
            for split, ids in split_ids.items():
                path = emb_dir / emb_cfg["file_template"].format(split=split)
                df = pd.read_parquet(path).set_index(self.id_col)
                columns = _embedding_columns(df, emb_cfg)
                cache[key][split] = df.loc[ids, columns]
        return cache

    def _embedding_feature_dims(
        self,
        cache: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Iterable[Tuple[str, int]]:
        for key, prefix in (("text_embedding", "text"), ("image_embedding", "image")):
            if key in cache:
                yield prefix, cache[key]["train"].shape[1]

    def _df_for_ids(self, split: str, ids: pd.Index) -> pd.DataFrame:
        return self.meta[split].set_index(self.id_col).loc[ids].reset_index()


def _embedding_columns(df: pd.DataFrame, emb_cfg: dict) -> List[str]:
    prefix = emb_cfg.get("prefix")
    if prefix:
        cols = [col for col in df.columns if str(col).startswith(prefix)]
        if cols:
            return cols
    return [col for col in df.columns if col != "id"]


def _parse_multi_value(value, parser: str) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    if parser == "literal_list":
        try:
            parsed = ast.literal_eval(str(value))
            return [str(item) for item in parsed]
        except Exception:
            return []
    if parser == "anilist_studios_json":
        try:
            parsed = json.loads(str(value))
            return [
                str(item["node"]["name"])
                for item in parsed
                if isinstance(item, dict) and "node" in item and "name" in item["node"]
            ]
        except Exception:
            return []
    if parser == "pipe":
        return [part.strip() for part in str(value).split("|") if part.strip()]
    raise ValueError(f"Unknown multihot parser: {parser}")

