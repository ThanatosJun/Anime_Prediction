"""
Train text-only baseline regressors from embedding artifacts.

This script:
1) Loads embedding parquet files for train/val/test
2) Runs ID overlap checks across splits
3) Trains two Ridge regressors (popularity, meanScore)
4) Evaluates MAE, RMSE, and Spearman on val/test
5) Saves metrics and run metadata to JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-only baseline regressors.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--embedding-prefix", type=str, default="text_embeddings")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["popularity", "meanScore"],
        help="Target columns to train/evaluate.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization alpha.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--report-name",
        type=str,
        default="text_branch_metrics.json",
        help="Output JSON report filename.",
    )
    return parser.parse_args()


def _embedding_file(artifact_dir: Path, prefix: str, split: str) -> Path:
    return artifact_dir / f"{prefix}_{split}.parquet"


def _load_split(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding artifact for {split}: {path}")
    return pd.read_parquet(path)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns found. Expected columns like emb_000, emb_001, ...")
    return emb_cols


def _split_overlap(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, id_column: str) -> Dict:
    for name, df in (("train", df_train), ("val", df_val), ("test", df_test)):
        if id_column not in df.columns:
            raise ValueError(f"Missing id column '{id_column}' in {name} split")

    train_ids = set(df_train[id_column].dropna().astype(str).tolist())
    val_ids = set(df_val[id_column].dropna().astype(str).tolist())
    test_ids = set(df_test[id_column].dropna().astype(str).tolist())

    train_val = train_ids.intersection(val_ids)
    train_test = train_ids.intersection(test_ids)
    val_test = val_ids.intersection(test_ids)

    return {
        "train_val_overlap": len(train_val),
        "train_test_overlap": len(train_test),
        "val_test_overlap": len(val_test),
        "no_overlap": len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0,
    }


def _prepare_xy(df: pd.DataFrame, feature_cols: List[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
    if target not in df.columns:
        raise ValueError(f"Missing target column '{target}'")

    work = df.loc[:, feature_cols + [target]].copy()
    work = work.dropna(subset=[target])

    x = work[feature_cols].to_numpy(dtype=np.float32)
    y = work[target].to_numpy(dtype=np.float32)

    if len(x) == 0:
        raise ValueError(f"No rows available after dropping NaN target for '{target}'")

    return x, y


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho, _ = spearmanr(y_true, y_pred)
    rho_value = float(rho) if np.isfinite(rho) else 0.0
    return {"MAE": mae, "RMSE": rmse, "Spearman": rho_value}


def _train_and_eval(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    alpha: float,
    random_seed: int,
) -> Dict:
    x_train, y_train = _prepare_xy(df_train, feature_cols, target)
    x_val, y_val = _prepare_xy(df_val, feature_cols, target)
    x_test, y_test = _prepare_xy(df_test, feature_cols, target)

    model = Ridge(alpha=alpha, random_state=random_seed)
    model.fit(x_train, y_train)

    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)

    return {
        "target": target,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "val": _evaluate(y_val, pred_val),
        "test": _evaluate(y_test, pred_test),
    }


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def main() -> None:
    args = _parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    train_path = _embedding_file(args.artifact_dir, args.embedding_prefix, "train")
    val_path = _embedding_file(args.artifact_dir, args.embedding_prefix, "val")
    test_path = _embedding_file(args.artifact_dir, args.embedding_prefix, "test")

    df_train = _load_split(train_path, "train")
    df_val = _load_split(val_path, "val")
    df_test = _load_split(test_path, "test")

    feature_cols = _feature_columns(df_train)

    overlap = _split_overlap(df_train, df_val, df_test, args.id_column)

    target_reports = {}
    for target in args.targets:
        target_reports[target] = _train_and_eval(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            feature_cols=feature_cols,
            target=target,
            alpha=args.alpha,
            random_seed=args.random_seed,
        )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": "Ridge",
            "alpha": args.alpha,
            "random_seed": args.random_seed,
        },
        "package_versions": {
            "python": sys.version.split()[0],
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scipy": _pkg_version("scipy"),
            "scikit-learn": _pkg_version("scikit-learn"),
            "pyarrow": _pkg_version("pyarrow"),
        },
        "inputs": {
            "train_path": str(train_path.as_posix()),
            "val_path": str(val_path.as_posix()),
            "test_path": str(test_path.as_posix()),
            "feature_count": len(feature_cols),
            "targets": args.targets,
        },
        "split_integrity": overlap,
        "metrics": target_reports,
    }

    out_path = args.report_dir / args.report_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics report: {out_path.as_posix()}")
    for target in args.targets:
        t = target_reports[target]
        print(
            f"{target} | val: MAE={t['val']['MAE']:.4f}, RMSE={t['val']['RMSE']:.4f}, Spearman={t['val']['Spearman']:.4f} | "
            f"test: MAE={t['test']['MAE']:.4f}, RMSE={t['test']['RMSE']:.4f}, Spearman={t['test']['Spearman']:.4f}"
        )


if __name__ == "__main__":
    main()
