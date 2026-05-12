from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


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
    if model_name == "cross_modal_transformer":
        return CrossModalTransformerRegressor(**params)
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


class CrossModalTransformerRegressor:
    """Small sklearn-style text/image transformer fusion regressor."""

    def __init__(
        self,
        text_dim: int = 384,
        image_dim: int = 1024,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 80,
        patience: int = 10,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device
        self.torch_num_threads = torch_num_threads
        self.verbose = verbose

    def fit(self, x, y):
        try:
            import torch
            from torch import nn
        except Exception as exc:
            raise ImportError("torch is not installed; install it or disable C2-CTNN-Lite") from exc

        if self.torch_num_threads:
            torch.set_num_threads(int(self.torch_num_threads))
        self._torch = torch
        self._nn = nn
        self._device = torch.device(self._resolve_device(torch))
        self._set_seed(torch)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        expected_dim = self.text_dim + self.image_dim
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"cross_modal_transformer expects {expected_dim} features "
                f"({self.text_dim} text + {self.image_dim} image), got {x.shape[1]}"
            )

        train_idx, val_idx = self._train_val_indices(len(x))
        self.x_scaler_ = StandardScaler().fit(x[train_idx])
        x_scaled = self.x_scaler_.transform(x).astype(np.float32)
        self.y_mean_ = float(y[train_idx].mean())
        y_std = float(y[train_idx].std())
        self.y_std_ = y_std if y_std > 1e-8 else 1.0
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        self.model_ = _make_cross_modal_transformer_net(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        x_train = torch.from_numpy(x_scaled[train_idx])
        y_train = torch.from_numpy(y_scaled[train_idx])
        x_val = torch.from_numpy(x_scaled[val_idx]).to(self._device)
        y_val = torch.from_numpy(y_scaled[val_idx]).to(self._device)

        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0
        rng = np.random.default_rng(self.random_state)
        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            order = rng.permutation(len(train_idx))
            batch_losses = []
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                xb = x_train[batch_idx].to(self._device)
                yb = y_train[batch_idx].to(self._device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(x_val), y_val).detach().cpu().item())
            if self.verbose:
                train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
                print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = deepcopy(self.model_.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_loss
        return self

    def predict(self, x):
        torch = self._torch
        x = np.asarray(x, dtype=np.float32)
        x_scaled = self.x_scaler_.transform(x).astype(np.float32)
        preds = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(x_scaled), self.batch_size):
                xb = torch.from_numpy(x_scaled[start : start + self.batch_size]).to(self._device)
                pred = self.model_(xb).detach().cpu().numpy().reshape(-1)
                preds.append(pred)
        y_scaled = np.concatenate(preds, axis=0)
        return y_scaled * self.y_std_ + self.y_mean_

    def _resolve_device(self, torch) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _set_seed(self, torch) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _train_val_indices(self, n_rows: int):
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n_rows)
        n_val = max(1, int(round(n_rows * self.validation_fraction)))
        val_idx = order[:n_val]
        train_idx = order[n_val:]
        return train_idx, val_idx


def _make_cross_modal_transformer_net(
    text_dim: int,
    image_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_proj = nn.Sequential(nn.Linear(text_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.image_proj = nn.Sequential(nn.Linear(image_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.modality_embedding = nn.Parameter(torch.zeros(1, 2, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            text = x[:, :text_dim]
            image = x[:, text_dim : text_dim + image_dim]
            tokens = torch.stack([self.text_proj(text), self.image_proj(image)], dim=1)
            fused = self.encoder(tokens + self.modality_embedding)
            return self.head(fused.mean(dim=1))

    return Net()
