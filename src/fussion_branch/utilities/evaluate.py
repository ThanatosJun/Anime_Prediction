"""
Evaluation metrics per pipeline doc:
  - MAE / RMSE            (original scale, interpretable)
  - MAPE                  (popularity: % error, scale-free)
  - log_MAE               (popularity: error in log space = what model actually optimizes)
  - Spearman ρ            (rank correlation, scale-free)
  - bucket_accuracy       (popularity: global quartile classification accuracy)
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
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho, _ = spearmanr(y_true, y_pred)

    metrics: Dict[str, float] = {
        "MAE":          round(mae,  4),
        "RMSE":         round(rmse, 4),
        "Spearman_rho": round(float(rho), 4),
    }

    if target_col == "popularity":
        # MAPE: % error — interpretable across all popularity magnitudes
        # clip true to avoid division by near-zero (popularity min=25, safe in practice)
        mape = float(np.mean(np.abs(y_true - y_pred) / np.clip(y_true, 1, None)) * 100)
        metrics["MAPE"] = round(mape, 4)

        # log-scale MAE: error in the space the model actually trains in
        log_mae = float(np.mean(np.abs(np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None)))))
        metrics["log_MAE"] = round(log_mae, 4)

        # bucket accuracy: global quartile classification from training set
        if train_meta_df is not None:
            q25, q50, q75 = np.percentile(
                train_meta_df["popularity"].dropna().values, [25, 50, 75]
            )

            def to_bucket(arr):
                return np.where(arr < q25, 0,
                       np.where(arr < q50, 1,
                       np.where(arr < q75, 2, 3)))

            acc = float(np.mean(to_bucket(y_true) == to_bucket(y_pred)))
            metrics["bucket_accuracy"] = round(acc, 4)

    return metrics


def denormalize(y_norm: np.ndarray, scaler: dict) -> np.ndarray:
    y = y_norm * scaler["std"] + scaler["mean"]
    if scaler.get("log_transform", False):
        y = np.expm1(y)
    return y
