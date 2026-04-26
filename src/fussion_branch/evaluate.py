"""
Evaluation metrics per pipeline doc:
  - MAE
  - RMSE
  - Spearman ρ  (rank correlation)
  - popularity: popularity_quarter_bucket accuracy (global quartile bins from training)
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: str,
    train_meta_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho, _ = spearmanr(y_true, y_pred)

    metrics: Dict[str, float] = {
        "MAE":  round(mae,  4),
        "RMSE": round(rmse, 4),
        "Spearman_rho": round(float(rho), 4),
    }

    # popularity bucket accuracy: classify predictions using training-set global percentiles
    if target_col == "popularity" and train_meta_df is not None:
        q25, q50, q75 = np.percentile(
            train_meta_df["popularity"].dropna().values, [25, 50, 75]
        )

        def to_bucket(arr):
            buckets = np.where(arr < q25, 0,
                      np.where(arr < q50, 1,
                      np.where(arr < q75, 2, 3)))
            return buckets

        acc = float(np.mean(to_bucket(y_true) == to_bucket(y_pred)))
        metrics["bucket_accuracy"] = round(acc, 4)

    return metrics


def denormalize(y_norm: np.ndarray, scaler: dict) -> np.ndarray:
    y = y_norm * scaler["std"] + scaler["mean"]
    if scaler.get("log_transform", False):
        y = np.expm1(y)
    return y
