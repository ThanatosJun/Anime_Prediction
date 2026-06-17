"""Run sample-aligned reference baselines for paper follow-up checks.

This script keeps the original reference baseline runner untouched. It reruns a
small paper-facing subset of baselines under two explicit evaluation policies:

- strict_common: intersect rows with every requested feature artifact.
- zero_fallback_full: keep the metadata split rows and zero-fill missing
  embedding/RAG artifacts, matching the proposed model's missing-modality
  evaluation policy more closely.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_common.features import (
    EMBEDDING_FEATURE_KEYS,
    EMBEDDING_FEATURE_PREFIXES,
    MetadataEncoder,
    RagFeatureEncoder,
    _embedding_columns,
    _embedding_dir_key,
)
from src.experiment_common.metrics import compute_metrics
from src.reference_baseline_branch.run_reference_baselines import (
    _resolve_model_params,
)
from src.reference_baseline_branch.sklearn_models import make_model


DEFAULT_BASELINES = [
    "F1-RF-Meta",
    "F2-XGB-Concat",
    "C3-RAG-Selective-XGB",
]
DEFAULT_TARGETS = ["popularity", "meanScore"]
DEFAULT_POLICIES = ["strict_common", "zero_fallback_full"]


@dataclass
class SplitData:
    ids: np.ndarray
    x: np.ndarray | None
    y_raw: np.ndarray


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_metadata(config: dict) -> Dict[str, pd.DataFrame]:
    data_cfg = config["data"]
    meta_dir = Path(data_cfg["meta_dir"])
    suffix = data_cfg.get("meta_suffix", "")
    return {
        split: pd.read_csv(meta_dir / f"fusion_meta_clean_{split}{suffix}.csv")
        for split in data_cfg.get("splits", ["train", "val", "test"])
    }


def _feature_file(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path).set_index(id_col)


def _ids_for_policy(
    config: dict,
    meta: Dict[str, pd.DataFrame],
    feature_set: dict,
    policy: str,
) -> Dict[str, pd.Index]:
    id_col = config["data"].get("id_col", "id")
    split_ids = {
        split: pd.Index(df[id_col].astype(int).values, name=id_col)
        for split, df in meta.items()
    }
    if policy == "zero_fallback_full":
        return split_ids
    if policy != "strict_common":
        raise ValueError(f"Unknown policy: {policy}")

    for key in EMBEDDING_FEATURE_KEYS:
        if not feature_set.get(key, False):
            continue
        emb_cfg = config["features"][key]
        emb_dir = Path(config["data"][_embedding_dir_key(key, emb_cfg)])
        template = emb_cfg["file_template"]
        for split in list(split_ids):
            path = emb_dir / template.format(split=split)
            emb_ids = pd.read_parquet(path, columns=[id_col])[id_col].astype(int)
            split_ids[split] = split_ids[split].intersection(pd.Index(emb_ids))

    if feature_set.get("rag", False):
        rag_dir = Path(feature_set["rag_features_dir"])
        for split in list(split_ids):
            path = rag_dir / f"rag_features_{split}.parquet"
            rag_ids = pd.read_parquet(path, columns=[id_col])[id_col].astype(int)
            split_ids[split] = split_ids[split].intersection(pd.Index(rag_ids))
    return split_ids


def _df_for_ids(meta: Dict[str, pd.DataFrame], split: str, ids: pd.Index, id_col: str) -> pd.DataFrame:
    return meta[split].set_index(id_col).loc[ids].reset_index()


def _load_embedding_part(
    config: dict,
    key: str,
    split: str,
    ids: pd.Index,
    policy: str,
) -> tuple[np.ndarray, List[str], int]:
    id_col = config["data"].get("id_col", "id")
    emb_cfg = config["features"][key]
    emb_dir = Path(config["data"][_embedding_dir_key(key, emb_cfg)])
    path = emb_dir / emb_cfg["file_template"].format(split=split)
    df = _feature_file(path, id_col)
    columns = _embedding_columns(df, emb_cfg)
    if policy == "zero_fallback_full":
        aligned = df.reindex(ids)[columns].fillna(0.0)
    else:
        aligned = df.loc[ids, columns]
    prefix = emb_cfg.get("feature_name_prefix", EMBEDDING_FEATURE_PREFIXES[key])
    names = [f"{prefix}:{i}" for i in range(len(columns))]
    missing = int(aligned.isna().any(axis=1).sum())
    return aligned.values.astype(np.float32), names, missing


def _load_rag_frames(
    feature_set: dict,
    split_ids: Dict[str, pd.Index],
    id_col: str,
    policy: str,
) -> Dict[str, pd.DataFrame]:
    if not feature_set.get("rag", False):
        return {}
    rag_dir = Path(feature_set["rag_features_dir"])
    out: Dict[str, pd.DataFrame] = {}
    for split, ids in split_ids.items():
        path = rag_dir / f"rag_features_{split}.parquet"
        df = _feature_file(path, id_col)
        if policy == "zero_fallback_full":
            out[split] = df.reindex(ids).reset_index()
        else:
            out[split] = df.loc[ids].reset_index()
    return out


def _build_split_data(
    config: dict,
    meta: Dict[str, pd.DataFrame],
    feature_set: dict,
    target: str,
    policy: str,
) -> tuple[Dict[str, SplitData], List[str], dict]:
    id_col = config["data"].get("id_col", "id")
    split_ids = _ids_for_policy(config, meta, feature_set, policy)
    train_df = _df_for_ids(meta, "train", split_ids["train"], id_col)
    feature_names: List[str] = []
    diagnostics = {
        "policy": policy,
        "n_by_split": {split: int(len(ids)) for split, ids in split_ids.items()},
        "zero_filled_embeddings": {},
    }

    metadata_encoder = None
    if feature_set.get("metadata", False):
        metadata_encoder = MetadataEncoder(config["features"]["metadata"]).fit(train_df)
        feature_names.extend([f"meta:{name}" for name in metadata_encoder.feature_names])

    rag_encoder = None
    rag_frames = _load_rag_frames(feature_set, split_ids, id_col, policy)
    if feature_set.get("rag", False):
        rag_cfg = config["features"][feature_set.get("rag_config", "rag")]
        rag_encoder = RagFeatureEncoder(rag_cfg).fit(rag_frames["train"])
        feature_names.extend([f"rag:{name}" for name in rag_encoder.feature_names])

    embedding_names: Dict[str, List[str]] = {}
    output: Dict[str, SplitData] = {}
    for split, ids in split_ids.items():
        df = _df_for_ids(meta, split, ids, id_col)
        parts: List[np.ndarray] = []
        if metadata_encoder is not None:
            parts.append(metadata_encoder.transform(df))
        if rag_encoder is not None:
            parts.append(rag_encoder.transform(rag_frames[split]))
        for key in EMBEDDING_FEATURE_KEYS:
            if not feature_set.get(key, False):
                continue
            arr, names, missing = _load_embedding_part(config, key, split, ids, policy)
            parts.append(arr)
            embedding_names.setdefault(key, names)
            diagnostics["zero_filled_embeddings"].setdefault(key, {})[split] = missing
        x = None if not parts else np.concatenate(parts, axis=1)
        output[split] = SplitData(
            ids=ids.to_numpy(dtype=np.int64),
            x=x,
            y_raw=df[target].values.astype(np.float64),
        )

    for key in EMBEDDING_FEATURE_KEYS:
        if key in embedding_names:
            feature_names.extend(embedding_names[key])
    diagnostics["n_features"] = int(0 if output["train"].x is None else output["train"].x.shape[1])
    return output, feature_names, diagnostics


def _transform_target(y: np.ndarray, log_transform: bool) -> np.ndarray:
    if log_transform:
        return np.log1p(np.clip(y, 0, None))
    return y.astype(np.float64)


def _inverse_target(y: np.ndarray, log_transform: bool) -> np.ndarray:
    if log_transform:
        return np.expm1(y)
    return y.astype(np.float64)


def _predict(model: object, model_name: str, data: SplitData) -> np.ndarray:
    if model_name == "mean":
        return model.predict(None)
    if data.x is None:
        raise ValueError("Feature matrix is required for non-mean model")
    return model.predict(data.x)


def _enabled_baselines(config: dict, ids: Sequence[str]) -> List[dict]:
    wanted = set(ids)
    return [baseline for baseline in config["baselines"] if baseline["id"] in wanted]


def _run_one(
    config: dict,
    meta: Dict[str, pd.DataFrame],
    baseline: dict,
    target: str,
    policy: str,
    out_root: Path,
) -> dict:
    feature_set = config["feature_sets"][baseline["feature_set"]]
    split_data, feature_names, diagnostics = _build_split_data(
        config=config,
        meta=meta,
        feature_set=feature_set,
        target=target,
        policy=policy,
    )
    model_name = baseline["model"]
    params = _resolve_model_params(baseline.get("params", {}), feature_names)
    model = make_model(model_name, params)
    log_transform = bool(config["targets"][target].get("log_transform", False))

    train = split_data["train"]
    y_train = _transform_target(train.y_raw, log_transform)
    if model_name == "mean":
        model.fit(None, y_train)
    else:
        model.fit(train.x, y_train)

    row = {
        "baseline_id": baseline["id"],
        "target": target,
        "policy": policy,
        "model": model_name,
        "feature_set": baseline["feature_set"],
        "n_train": int(len(train.y_raw)),
        "n_features": diagnostics["n_features"],
        "diagnostics": json.dumps(diagnostics, sort_keys=True),
    }
    pred_dir = out_root / "predictions" / policy / baseline["id"] / target
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split in ["val", "test"]:
        data = split_data[split]
        pred = _inverse_target(_predict(model, model_name, data), log_transform)
        metrics = compute_metrics(data.y_raw, pred, target)
        for key, value in metrics.items():
            row[f"{split}_{key}"] = value
        row[f"n_{split}"] = int(len(data.y_raw))
        pd.DataFrame({"id": data.ids, "target": data.y_raw, "prediction": pred}).to_csv(
            pred_dir / f"{split}_predictions.csv",
            index=False,
        )
    with (pred_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)
    return row


def _write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "policy",
        "baseline_id",
        "target",
        "n_test",
        "test_log_MAE",
        "test_factor_acc_2x",
        "test_Spearman_rho",
        "test_MAE",
        "test_acc_within_10pt",
        "test_R2",
    ]
    present = [col for col in cols if col in summary.columns]
    lines = [
        "# Sample-aligned baseline evaluation",
        "",
        "Policies:",
        "",
        "- `strict_common`: uses only rows with all requested artifacts.",
        "- `zero_fallback_full`: keeps metadata split rows and zero-fills missing artifact vectors.",
        "",
        summary[present].to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
    )
    parser.add_argument("--output-dir", default="reports/experiments/sample_alignment")
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    meta = _load_metadata(config)
    baselines = _enabled_baselines(config, args.baselines)
    missing = sorted(set(args.baselines) - {baseline["id"] for baseline in baselines})
    if missing:
        raise ValueError(f"Missing baselines in config: {missing}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in args.policies:
        for baseline in baselines:
            for target in args.targets:
                print(f"[sample-align] {policy} {baseline['id']} {target}")
                rows.append(_run_one(config, meta, baseline, target, policy, out_root))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "sample_aligned_baseline_metrics.csv", index=False)
    _write_markdown(summary, out_root / "sample_aligned_baseline_metrics.md")


if __name__ == "__main__":
    main()
