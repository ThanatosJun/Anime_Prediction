from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


class MeanRegressor:
    def fit(self, x, y):
        self.value_ = float(np.mean(y))
        return self

    def predict_n(self, n_rows: int):
        return np.full(n_rows, self.value_, dtype=np.float64)


def make_model(model_name: str, params: dict):
    if model_name == "mean":
        return MeanRegressor()
    if model_name == "ridge":
        return Ridge(**params)
    if model_name == "random_forest":
        return RandomForestRegressor(**params)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params)
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    if model_name == "mlp":
        return MLPRegressor(**params)
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("xgboost is not installed; install it or disable F2-XGB-Concat") from exc
        return XGBRegressor(**params)
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise ImportError("lightgbm is not installed; install it or disable this baseline") from exc
        return LGBMRegressor(**params)
    raise ValueError(f"Unknown baseline model: {model_name}")

