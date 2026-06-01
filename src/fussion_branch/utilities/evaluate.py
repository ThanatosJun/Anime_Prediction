"""
Evaluation metrics:
  Both targets (primary → supporting)
  - Spearman_rho     rank correlation
  - R2               coefficient of determination (diagnoses distribution shift)
  - MAE              absolute error in original scale

  popularity only
  - log_MAE          MAE in log1p space — scale-free, matches training objective
"""
from typing import Dict

import numpy as np
from scipy.stats import spearmanr


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_col: str,
) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rho, _ = spearmanr(y_true, y_pred)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    metrics: Dict[str, float] = {
        "Spearman_rho": round(float(rho), 4),
        "R2":           round(r2,  4),
        "MAE":          round(mae, 4),
    }

    if target_col == "popularity":
        y_true_log = np.log1p(np.clip(y_true, 0, None))
        y_pred_log = np.log1p(np.clip(y_pred, 0, None))
        log_mae = float(np.mean(np.abs(y_true_log - y_pred_log)))
        log_ss_res = float(np.sum((y_true_log - y_pred_log) ** 2))
        log_ss_tot = float(np.sum((y_true_log - np.mean(y_true_log)) ** 2))
        log_r2 = 1.0 - log_ss_res / log_ss_tot if log_ss_tot > 0 else 0.0
        factor_acc_2x = float(np.mean(np.abs(y_true_log - y_pred_log) < np.log(2.0)))
        metrics["log_MAE"] = round(log_mae, 4)
        metrics["log_R2"] = round(float(log_r2), 4)
        metrics["factor_acc_2x"] = round(factor_acc_2x, 4)
    elif target_col == "meanScore":
        acc_within_10pt = float(np.mean(np.abs(y_true - y_pred) < 10.0))
        metrics["acc_within_10pt"] = round(acc_within_10pt, 4)

    return metrics


def denormalize(y_norm: np.ndarray, scaler: dict) -> np.ndarray:
    y_norm = np.asarray(y_norm, dtype=np.float64)   # model may output float16 under AMP
    if scaler.get("log_transform", False):
        # Clip in normalized space before expm1: predictions beyond ±5σ are
        # physically implausible; float16 overflows expm1 for input > ~10.8.
        y_norm = np.clip(y_norm, -5.0, 5.0)
    center = scaler.get("center", scaler.get("mean", 0.0))
    scale  = scaler.get("scale",  scaler.get("std",  1.0))
    y = y_norm * float(scale) + float(center)
    if scaler.get("log_transform", False):
        y = np.expm1(y)
    return y
