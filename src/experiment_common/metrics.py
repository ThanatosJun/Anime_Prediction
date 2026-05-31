from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr


def inverse_target(y: np.ndarray, log_transform: bool) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if log_transform:
        return np.expm1(y)
    return y


def transform_target(y: np.ndarray, log_transform: bool) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if log_transform:
        return np.log1p(np.clip(y, 0, None))
    return y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rho = _safe_corr(lambda a, b: spearmanr(a, b).statistic, y_true, y_pred)
    pearson = _safe_corr(lambda a, b: pearsonr(a, b).statistic, y_true, y_pred)

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(float(r2), 4),
        "Spearman_rho": round(rho, 4),
        "Pearson_r": round(pearson, 4),
    }
    if target == "popularity":
        log_mae = float(
            np.mean(
                np.abs(
                    np.log1p(np.clip(y_true, 0, None))
                    - np.log1p(np.clip(y_pred, 0, None))
                )
            )
        )
        metrics["log_MAE"] = round(log_mae, 4)
    return metrics


def _safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return 0.0
    value = float(fn(y_true, y_pred))
    if np.isnan(value):
        return 0.0
    return value

