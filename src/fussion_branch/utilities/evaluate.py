"""
Evaluation metrics (both targets):
  - MSE / RMSE       original scale
  - MAE              original scale
  - MAE_over_median  MAE normalised by training-set median (scale-free, comparable across targets)
  - Spearman_rho     rank correlation
  - R2               coefficient of determination
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
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    rho, _ = spearmanr(y_true, y_pred)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    metrics: Dict[str, float] = {
        "MSE":          round(mse,  4),
        "RMSE":         round(rmse, 4),
        "MAE":          round(mae,  4),
        "Spearman_rho": round(float(rho), 4),
        "R2":           round(r2,   4),
    }

    if train_meta_df is not None and target_col in train_meta_df.columns:
        median_val = float(np.median(train_meta_df[target_col].dropna().values))
        metrics["MAE_over_median"] = round(mae / max(median_val, 1e-8), 4)

    return metrics


def denormalize(y_norm: np.ndarray, scaler: dict) -> np.ndarray:
    y_norm = np.asarray(y_norm, dtype=np.float64)   # model may output float16 under AMP
    if scaler.get("log_transform", False):
        # Clip in normalized space before expm1: predictions beyond ±5σ are
        # physically implausible; float16 overflows expm1 for input > ~10.8.
        y_norm = np.clip(y_norm, -5.0, 5.0)
    y = y_norm * float(scaler["std"]) + float(scaler["mean"])
    if scaler.get("log_transform", False):
        y = np.expm1(y)
    return y
