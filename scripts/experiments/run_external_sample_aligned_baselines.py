"""Run CARMA-input baselines on MAL 2025 external rows.

The paper reference baselines use older project embedding artifacts. The MAL
external exam, however, is already prepared through the CARMA input pipeline.
This script therefore trains lightweight baseline families on the internal
CARMA-input train split, then evaluates them on the exact MAL local-ready rows
used by the CARMA external inference files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_common.features import MetadataEncoder, RagFeatureEncoder
from src.experiment_common.metrics import compute_metrics
from src.reference_baseline_branch.run_reference_baselines import _resolve_model_params
from src.reference_baseline_branch.sklearn_models import make_model


BASELINES = {
    "F1-RF-Meta": {
        "model": "random_forest",
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": 1,
        },
        "features": ["metadata"],
    },
    "F2-XGB-Concat-CARMAInput": {
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
        "features": ["metadata", "text", "image"],
    },
    "C3-RAG-XGB-CARMAInput": {
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
        "features": ["metadata", "text", "image", "rag"],
    },
}

EXAMS = {
    "mal2025_popularity_local_ready": {
        "meta": "src_2/data/dataset/fusion_meta_clean_mal2025_popularity_local_ready_v2.csv",
        "targets": ["popularity"],
        "carma_metrics": "data/external_transformed/run02_mal2025_popularity_local_ready_metrics.json",
    },
    "mal2025_dual_local_ready": {
        "meta": "src_2/data/dataset/fusion_meta_clean_mal2025_dual_local_ready_v2.csv",
        "targets": ["popularity", "meanScore"],
        "carma_metrics": "data/external_transformed/run02_mal2025_dual_local_ready_metrics.json",
    },
}


@dataclass
class MatrixBundle:
    ids: np.ndarray
    x: np.ndarray | None
    y: np.ndarray


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_parquet_matrix(path: Path, ids: pd.Index, prefixes: Sequence[str] | None = None) -> np.ndarray:
    df = pd.read_parquet(path).set_index("id")
    if prefixes:
        cols = [
            col
            for col in df.columns
            if any(str(col).startswith(prefix) for prefix in prefixes)
        ]
    else:
        cols = list(df.columns)
    return df.reindex(ids)[cols].fillna(0.0).values.astype(np.float32)


def _prepare_rag_frame(path: Path, ids: pd.Index) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("id").reindex(ids).reset_index()
    defaults = {
        "rag_title_romaji": None,
        "rag_popularity": 0.0,
        "rag_score": 0.0,
        "rag_release_year": 0.0,
        "rag_episodes": 0.0,
        "rag_similarity_mean": 0.0,
        "rag_similarity_max": 0.0,
        "rag_topk_count": 0.0,
        "rag_genres": "[]",
        "rag_format": None,
        "rag_studios": "[]",
        "rag_found": False,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        elif default is None:
            df[col] = df[col].where(df[col].notna(), None)
        else:
            df[col] = df[col].fillna(default)
    return df


def _feature_names(prefix: str, width: int) -> List[str]:
    return [f"{prefix}:{idx}" for idx in range(width)]


def _build_matrices(
    cfg: dict,
    train_meta: pd.DataFrame,
    eval_meta: pd.DataFrame,
    features: Sequence[str],
    target: str,
    eval_split: str,
) -> tuple[MatrixBundle, MatrixBundle, List[str], dict]:
    train_ids = pd.Index(train_meta["id"].astype(int).values, name="id")
    eval_ids = pd.Index(eval_meta["id"].astype(int).values, name="id")
    parts_train: List[np.ndarray] = []
    parts_eval: List[np.ndarray] = []
    names: List[str] = []
    diagnostics = {"zero_fallback": {}, "feature_dims": {}}

    if "metadata" in features:
        encoder = MetadataEncoder(cfg["features"]["metadata"]).fit(train_meta)
        train_part = encoder.transform(train_meta)
        eval_part = encoder.transform(eval_meta)
        parts_train.append(train_part)
        parts_eval.append(eval_part)
        names.extend([f"meta:{name}" for name in encoder.feature_names])
        diagnostics["feature_dims"]["metadata"] = int(train_part.shape[1])

    if "rag" in features:
        rag_cfg = cfg["features"]["rag"]
        train_rag = _prepare_rag_frame(Path("src_2/RAG/return/rag_features_train.parquet"), train_ids)
        eval_rag = _prepare_rag_frame(Path(f"src_2/RAG/return/rag_features_{eval_split}.parquet"), eval_ids)
        encoder = RagFeatureEncoder(rag_cfg).fit(train_rag)
        train_part = encoder.transform(train_rag)
        eval_part = encoder.transform(eval_rag)
        parts_train.append(train_part)
        parts_eval.append(eval_part)
        names.extend([f"rag:{name}" for name in encoder.feature_names])
        diagnostics["feature_dims"]["rag"] = int(train_part.shape[1])

    if "text" in features:
        train_part = _load_parquet_matrix(
            Path("src_2/embedding/text/text_embeddings_train.parquet"),
            train_ids,
            prefixes=["emb_"],
        )
        eval_part = _load_parquet_matrix(
            Path(f"src_2/embedding/text/text_embeddings_{eval_split}.parquet"),
            eval_ids,
            prefixes=["emb_"],
        )
        parts_train.append(train_part)
        parts_eval.append(eval_part)
        names.extend(_feature_names("text", train_part.shape[1]))
        diagnostics["feature_dims"]["text"] = int(train_part.shape[1])
        diagnostics["zero_fallback"]["text_train_rows"] = int((np.abs(train_part).sum(axis=1) == 0).sum())
        diagnostics["zero_fallback"]["text_eval_rows"] = int((np.abs(eval_part).sum(axis=1) == 0).sum())

    if "image" in features:
        train_part = _load_parquet_matrix(
            Path("src_2/embedding/image/image_embeddings_train.parquet"),
            train_ids,
            prefixes=["yolo_", "cover_", "banner_"],
        )
        eval_part = _load_parquet_matrix(
            Path(f"src_2/embedding/image/image_embeddings_{eval_split}.parquet"),
            eval_ids,
            prefixes=["yolo_", "cover_", "banner_"],
        )
        parts_train.append(train_part)
        parts_eval.append(eval_part)
        names.extend(_feature_names("image", train_part.shape[1]))
        diagnostics["feature_dims"]["image"] = int(train_part.shape[1])
        diagnostics["zero_fallback"]["image_train_rows"] = int((np.abs(train_part).sum(axis=1) == 0).sum())
        diagnostics["zero_fallback"]["image_eval_rows"] = int((np.abs(eval_part).sum(axis=1) == 0).sum())

    x_train = None if not parts_train else np.concatenate(parts_train, axis=1)
    x_eval = None if not parts_eval else np.concatenate(parts_eval, axis=1)
    return (
        MatrixBundle(train_ids.to_numpy(), x_train, train_meta[target].values.astype(np.float64)),
        MatrixBundle(eval_ids.to_numpy(), x_eval, eval_meta[target].values.astype(np.float64)),
        names,
        diagnostics,
    )


def _transform(y: np.ndarray, target: str) -> np.ndarray:
    if target == "popularity":
        return np.log1p(np.clip(y, 0, None))
    return y.astype(np.float64)


def _inverse(y: np.ndarray, target: str) -> np.ndarray:
    if target == "popularity":
        return np.expm1(y)
    return y.astype(np.float64)


def _carma_rows() -> List[dict]:
    rows: List[dict] = []
    for exam, info in EXAMS.items():
        path = Path(info["carma_metrics"])
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        if "popularity" in metrics:
            pop = metrics["popularity"]
            rows.append(
                {
                    "model": f"CARMA-Run{payload['run_id']}",
                    "exam": exam,
                    "target": "popularity",
                    "n": pop["n"],
                    "Spearman_rho": pop["spearman_prediction_vs_mal_members"],
                    "log_MAE": pop["log_mae_prediction_vs_mal_members"],
                    "log_R2": pop["log_r2_prediction_vs_mal_members"],
                    "factor_acc_2x": pop["factor_acc_2x_prediction_vs_mal_members"],
                    "MAE": pop["raw_mae_prediction_vs_mal_members"],
                    "R2": "",
                    "acc_within_10pt": "",
                    "notes": "existing external inference artifact",
                }
            )
        if "meanScore" in metrics:
            score = metrics["meanScore"]
            rows.append(
                {
                    "model": f"CARMA-Run{payload['run_id']}",
                    "exam": exam,
                    "target": "meanScore",
                    "n": score["n"],
                    "Spearman_rho": score["spearman"],
                    "MAE": score["mae"],
                    "R2": score["r2"],
                    "acc_within_10pt": score["acc_within_10pt"],
                    "log_MAE": "",
                    "log_R2": "",
                    "factor_acc_2x": "",
                    "notes": "existing external inference artifact",
                }
            )
    return rows


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        "exam",
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
    lines = [
        "# External MAL sample-aligned evaluation",
        "",
        "These baselines use CARMA-input artifacts so they can be evaluated on the exact MAL local-ready rows.",
        "They are not exact replacements for the older paper reference baselines that used 384-d text and 1024-d image artifacts.",
        "",
        summary[cols].to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/reference_baseline_branch/configs/reference_baselines.yaml")
    parser.add_argument("--output-dir", default="reports/experiments/sample_alignment")
    parser.add_argument("--include-carma", action="store_true", default=True)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "external_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    train_meta = pd.read_csv("src_2/data/dataset/fusion_meta_clean_train_v2.csv")
    rows: List[dict] = []
    if args.include_carma:
        rows.extend(_carma_rows())

    for exam, exam_info in EXAMS.items():
        eval_meta = pd.read_csv(exam_info["meta"])
        eval_split = exam
        for baseline_id, spec in BASELINES.items():
            for target in exam_info["targets"]:
                print(f"[external-align] {exam} {baseline_id} {target}")
                train, eval_bundle, feature_names, diagnostics = _build_matrices(
                    cfg=cfg,
                    train_meta=train_meta,
                    eval_meta=eval_meta,
                    features=spec["features"],
                    target=target,
                    eval_split=eval_split,
                )
                model = make_model(spec["model"], _resolve_model_params(spec["params"], feature_names))
                model.fit(train.x, _transform(train.y, target))
                pred = _inverse(model.predict(eval_bundle.x), target)
                metrics = compute_metrics(eval_bundle.y, pred, target)
                out = pred_dir / f"{exam}__{baseline_id}__{target}.csv"
                pd.DataFrame(
                    {"id": eval_bundle.ids, "target": eval_bundle.y, "prediction": pred}
                ).to_csv(out, index=False)
                rows.append(
                    {
                        "model": baseline_id,
                        "exam": exam,
                        "target": target,
                        "n": int(len(eval_bundle.y)),
                        **metrics,
                        "notes": json.dumps(diagnostics, sort_keys=True),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "external_sample_aligned_metrics.csv", index=False)
    _write_markdown(summary, out_dir / "external_sample_aligned_metrics.md")


if __name__ == "__main__":
    main()
