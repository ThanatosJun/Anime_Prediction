"""Run baselines on flattened CARMA AnimeDataset tensors.

This is the strictest sample-aligned baseline adapter in this branch: it does
not rebuild embeddings and does not reimplement CARMA feature assembly. Instead,
it instantiates src_2.fussion_training.dataset.AnimeDataset and flattens the
actual tensors that CARMA would feed to FusionModel.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC2 = ROOT / "src_2"
if str(SRC2 / "fussion_training") not in sys.path:
    sys.path.insert(0, str(SRC2 / "fussion_training"))

from src.experiment_common.metrics import compute_metrics
from src.reference_baseline_branch.sklearn_models import make_model
from src_2.fussion_training.dataset import AnimeDataset, _build_target_scaler, denormalize_target
from src_2.fussion_training.meta_encoder import MetaEncoder


BASELINES = {
    "F1-RF-Meta-CARMATensor": {
        "model": "random_forest",
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": 1,
        },
        "groups": ["meta"],
    },
    "F2-XGB-Concat-CARMATensor": {
        "model": "xgboost",
        "params": {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "objective": "reg:squarederror",
        },
        "groups": ["meta", "text", "image"],
    },
    "C3-RAG-XGB-CARMATensor": {
        "model": "xgboost",
        "params": {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "objective": "reg:squarederror",
        },
        "groups": ["meta", "text", "image", "rag"],
    },
}

EXTERNAL_SPLITS = {
    "mal2025_popularity_local_ready": ["popularity"],
    "mal2025_dual_local_ready": ["popularity", "meanScore"],
}


@dataclass
class MatrixBundle:
    ids: np.ndarray
    x: np.ndarray
    y_raw: np.ndarray
    diagnostics: dict


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fit_meta_encoder(config: dict) -> MetaEncoder:
    meta_dir = Path(config["data"]["meta_dir"])
    suffix = config["data"].get("meta_suffix", "_v2")
    train_df = pd.read_csv(meta_dir / f"fusion_meta_clean_train{suffix}.csv")
    encoder = MetaEncoder()
    encoder.fit(train_df)
    return encoder


def _target_scaler(config: dict, meta_encoder: MetaEncoder, target: str) -> dict:
    meta_dir = Path(config["data"]["meta_dir"])
    suffix = config["data"].get("meta_suffix", "_v2")
    train_df = pd.read_csv(meta_dir / f"fusion_meta_clean_train{suffix}.csv")
    target_vals = pd.to_numeric(train_df[target], errors="coerce").fillna(0).values
    t_cfg = config.get("training", {}).get(target, {})
    return _build_target_scaler(
        target_vals,
        log_transform=t_cfg.get("log_transform", False),
        winsor_pct=t_cfg.get("winsor_pct", 100),
    )


def _denormalize_target(values: np.ndarray, scaler: dict) -> np.ndarray:
    return denormalize_target(values, scaler)


def _tensor_to_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_item(item: dict, groups: Sequence[str]) -> np.ndarray:
    parts: List[np.ndarray] = []
    if "meta" in groups:
        parts.append(_tensor_to_np(item["meta_feat"]).reshape(-1).astype(np.float32))
    if "text" in groups:
        parts.append(_tensor_to_np(item["text_emb"]).reshape(-1).astype(np.float32))
    if "image" in groups:
        image = _tensor_to_np(item["image_emb"]).reshape(-1).astype(np.float32)
        image_mask = _tensor_to_np(item["image_mask"]).astype(np.float32).reshape(-1)
        parts.extend([image, image_mask])
    if "rag" in groups:
        rag_meta = _tensor_to_np(item["rag_meta"]).reshape(-1).astype(np.float32)
        rag_text = _tensor_to_np(item["rag_text"]).reshape(-1).astype(np.float32)
        rag_image = _tensor_to_np(item["rag_image"]).reshape(-1).astype(np.float32)
        rag_mask = _tensor_to_np(item["rag_mask"]).astype(np.float32).reshape(-1)
        parts.extend([rag_meta, rag_text, rag_image, rag_mask])
    return np.concatenate(parts).astype(np.float32)


def _feature_names(ds: AnimeDataset, groups: Sequence[str]) -> List[str]:
    names: List[str] = []
    if "meta" in groups:
        names.extend([f"meta:{i}" for i in range(56)])
    if "text" in groups:
        names.extend([f"text:{i}" for i in range(ds.text_dim)])
    if "image" in groups:
        for modality in range(ds.n_image_modality):
            names.extend([f"image:{modality}:{i}" for i in range(ds.image_dim)])
        names.extend([f"image_mask:{i}" for i in range(ds.n_image_modality)])
    if "rag" in groups:
        names.extend([f"rag_meta:{i}" for i in range(ds.top_k * 10)])
        names.extend([f"rag_text:{i}" for i in range(ds.top_k * ds.text_dim)])
        names.extend([f"rag_image:{i}" for i in range(ds.top_k * ds.rag_image_dim)])
        names.extend([f"rag_mask:{i}" for i in range(ds.top_k)])
    return names


def _build_bundle(
    split: str,
    config: dict,
    meta_encoder: MetaEncoder,
    target: str,
    scaler: dict,
    groups: Sequence[str],
) -> tuple[MatrixBundle, List[str]]:
    ds = AnimeDataset(split, config, meta_encoder, target=target, target_scaler=scaler)
    rows: List[np.ndarray] = []
    ids: List[int] = []
    norm_targets: List[float] = []
    zero_text = 0
    all_image_missing = 0
    all_rag_missing = 0
    for idx in range(len(ds)):
        item = ds[idx]
        rows.append(_flatten_item(item, groups))
        ids.append(int(item["id"]))
        norm_targets.append(float(item["target"]))
        if "text" in groups and float(torch.abs(item["text_emb"]).sum()) == 0.0:
            zero_text += 1
        if "image" in groups and bool(item["image_mask"].all()):
            all_image_missing += 1
        if "rag" in groups and bool(item["rag_mask"].all()):
            all_rag_missing += 1
    x = np.vstack(rows).astype(np.float32)
    y_raw = _denormalize_target(np.asarray(norm_targets, dtype=np.float64), scaler)
    diagnostics = {
        "split": split,
        "groups": list(groups),
        "n_rows": len(ds),
        "n_features": int(x.shape[1]),
        "zero_text_rows": zero_text,
        "all_image_missing_rows": all_image_missing,
        "all_rag_missing_rows": all_rag_missing,
        "text_dim": int(ds.text_dim),
        "image_dim": int(ds.image_dim),
        "n_image_modality": int(ds.n_image_modality),
        "rag_image_dim": int(getattr(ds, "rag_image_dim", ds.image_dim)),
        "top_k": int(ds.top_k),
    }
    return MatrixBundle(np.asarray(ids, dtype=np.int64), x, y_raw, diagnostics), _feature_names(ds, groups)


def _transform_target(y: np.ndarray, target: str) -> np.ndarray:
    if target == "popularity":
        return np.log1p(np.clip(y, 0, None))
    return y.astype(np.float64)


def _inverse_target(y: np.ndarray, target: str) -> np.ndarray:
    if target == "popularity":
        return np.expm1(y)
    return y.astype(np.float64)


def _carma_external_rows() -> List[dict]:
    paths = [
        Path("data/external_transformed/run02_mal2025_popularity_local_ready_metrics.json"),
        Path("data/external_transformed/run02_mal2025_dual_local_ready_metrics.json"),
        Path("data/external_transformed/run22_mal2025_popularity_local_ready_metrics.json"),
        Path("data/external_transformed/run22_mal2025_dual_local_ready_metrics.json"),
    ]
    rows: List[dict] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        exam = payload["split"]
        run_id = payload["run_id"]
        metrics = payload["metrics"]
        if "popularity" in metrics:
            pop = metrics["popularity"]
            rows.append(
                {
                    "model": f"CARMA-Run{run_id}",
                    "split": exam,
                    "target": "popularity",
                    "n": pop["n"],
                    "MAE": pop["raw_mae_prediction_vs_mal_members"],
                    "Spearman_rho": pop["spearman_prediction_vs_mal_members"],
                    "log_MAE": pop["log_mae_prediction_vs_mal_members"],
                    "log_R2": pop["log_r2_prediction_vs_mal_members"],
                    "factor_acc_2x": pop["factor_acc_2x_prediction_vs_mal_members"],
                    "R2": "",
                    "acc_within_10pt": "",
                    "diagnostics": "existing CARMA Run02 external inference artifact",
                }
            )
        if "meanScore" in metrics:
            score = metrics["meanScore"]
            rows.append(
                {
                    "model": f"CARMA-Run{run_id}",
                    "split": exam,
                    "target": "meanScore",
                    "n": score["n"],
                    "MAE": score["mae"],
                    "R2": score["r2"],
                    "Spearman_rho": score["spearman"],
                    "acc_within_10pt": score["acc_within_10pt"],
                    "log_MAE": "",
                    "log_R2": "",
                    "factor_acc_2x": "",
                    "diagnostics": "existing CARMA Run02 external inference artifact",
                }
            )
    return rows


def _run_one(
    baseline_id: str,
    spec: dict,
    target: str,
    train: MatrixBundle,
    eval_bundle: MatrixBundle,
    split: str,
    out_dir: Path,
) -> dict:
    model = make_model(spec["model"], spec["params"])
    model.fit(train.x, _transform_target(train.y_raw, target))
    pred = _inverse_target(model.predict(eval_bundle.x), target)
    metrics = compute_metrics(eval_bundle.y_raw, pred, target)
    pred_dir = _prediction_dir(out_dir, split, baseline_id, target)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"id": eval_bundle.ids, "target": eval_bundle.y_raw, "prediction": pred}
    ).to_csv(pred_dir / "predictions.csv", index=False)
    return {
        "model": baseline_id,
        "split": split,
        "target": target,
        "n": int(len(eval_bundle.y_raw)),
        **metrics,
        "diagnostics": json.dumps(
            {"train": train.diagnostics, "eval": eval_bundle.diagnostics},
            sort_keys=True,
        ),
    }


def _prediction_dir(out_dir: Path, split: str, baseline_id: str, target: str) -> Path:
    return out_dir / "carma_tensor_predictions" / split / baseline_id / target


def _existing_prediction_row(
    out_dir: Path,
    split: str,
    baseline_id: str,
    target: str,
) -> dict | None:
    path = _prediction_dir(out_dir, split, baseline_id, target) / "predictions.csv"
    if not path.exists():
        return None
    pred_df = pd.read_csv(path)
    metrics = compute_metrics(
        pred_df["target"].to_numpy(dtype=np.float64),
        pred_df["prediction"].to_numpy(dtype=np.float64),
        target,
    )
    return {
        "model": baseline_id,
        "split": split,
        "target": target,
        "n": int(len(pred_df)),
        **metrics,
        "diagnostics": "resumed from existing CARMA tensor prediction file",
    }


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        "split",
        "model",
        "target",
        "n",
        "log_MAE",
        "factor_acc_2x",
        "Spearman_rho",
        "MAE",
        "acc_within_10pt",
        "R2",
    ]
    present = [col for col in cols if col in summary.columns]
    lines = [
        "# CARMA Tensor-aligned Baseline Evaluation",
        "",
        "Baselines in this table flatten the actual tensors returned by `src_2.fussion_training.dataset.AnimeDataset`.",
        "No text/image/RAG embeddings are regenerated.",
        "",
        summary[present].replace({np.nan: ""}).to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(rows: List[dict], out_dir: Path) -> None:
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "carma_tensor_aligned_metrics.csv", index=False)
    _write_markdown(summary, out_dir / "carma_tensor_aligned_metrics.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src_2/fussion_configs.yaml")
    parser.add_argument("--output-dir", default="reports/experiments/sample_alignment")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "mal2025_popularity_local_ready", "mal2025_dual_local_ready"],
    )
    parser.add_argument("--targets", nargs="+", default=["popularity", "meanScore"])
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES))
    parser.add_argument("--no-resume-existing", action="store_true")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    # Keep evaluation deterministic and CPU-friendly for dataset construction.
    config = copy.deepcopy(config)
    config.setdefault("training", {})["device"] = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_encoder = _fit_meta_encoder(config)
    rows: List[dict] = []
    rows.extend(_carma_external_rows())
    _write_outputs(rows, out_dir)

    bundle_cache: Dict[tuple[str, str, tuple[str, ...]], tuple[MatrixBundle, List[str]]] = {}
    for baseline_id in args.baselines:
        spec = BASELINES[baseline_id]
        groups = tuple(spec["groups"])
        for target in args.targets:
            scaler = _target_scaler(config, meta_encoder, target)
            train_key = ("train", target, groups)
            for split in args.splits:
                if split in EXTERNAL_SPLITS and target not in EXTERNAL_SPLITS[split]:
                    continue
                if split == "test" or split in EXTERNAL_SPLITS:
                    if not args.no_resume_existing:
                        existing = _existing_prediction_row(out_dir, split, baseline_id, target)
                        if existing is not None:
                            print(f"[carma-tensor-align] resumed {split} {baseline_id} {target}")
                            rows.append(existing)
                            _write_outputs(rows, out_dir)
                            continue
                    if train_key not in bundle_cache:
                        bundle_cache[train_key] = _build_bundle("train", config, meta_encoder, target, scaler, groups)
                    train, _ = bundle_cache[train_key]
                    key = (split, target, groups)
                    if key not in bundle_cache:
                        bundle_cache[key] = _build_bundle(split, config, meta_encoder, target, scaler, groups)
                    eval_bundle, _ = bundle_cache[key]
                    print(f"[carma-tensor-align] {split} {baseline_id} {target}")
                    rows.append(_run_one(baseline_id, spec, target, train, eval_bundle, split, out_dir))
                    _write_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
