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
    if model_name == "branch_mlp":
        return BranchMLPRegressor(**params)
    if model_name == "armenta_project_input_mlp":
        return ArmentaProjectInputMLPRegressor(**params)
    if model_name == "armenta_figure2_reconstruction":
        return ArmentaFigure2ReconstructionRegressor(**params)
    if model_name == "cross_modal_transformer":
        return CrossModalTransformerRegressor(**params)
    if model_name == "project_input_cross_attention":
        return ProjectInputCrossAttentionRegressor(**params)
    if model_name == "project_input_recurrent_fusion":
        return ProjectInputRecurrentFusionRegressor(**params)
    if model_name == "project_input_ctnn_reconstruction":
        return ProjectInputCTNNReconstructionRegressor(**params)
    if model_name == "project_input_ctnn_dual_visual_reconstruction":
        return ProjectInputCTNNDualVisualReconstructionRegressor(**params)
    if model_name == "project_input_skapp_graph_proxy":
        return ProjectInputSKAPPGraphProxyRegressor(**params)
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


class ProjectInputCrossAttentionRegressor:
    """Project-input CTNN-style proxy with explicit cross-attention and metadata-conditioned fusion."""

    model_label = "project_input_cross_attention"
    baseline_label = "C2-ProjectInputCrossAttention"

    def __init__(
        self,
        metadata_dim: int = 151,
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
        max_epochs: int = 90,
        patience: int = 10,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        self.metadata_dim = metadata_dim
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
            raise ImportError(f"torch is not installed; install it or disable {self.baseline_label}") from exc

        if self.torch_num_threads:
            torch.set_num_threads(int(self.torch_num_threads))
        self._torch = torch
        self._device = torch.device(self._resolve_device(torch))
        self._set_seed(torch)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        expected_dim = self._expected_dim()
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"{self.model_label} expects {expected_dim} features "
                f"({self._feature_shape_label()}), "
                f"got {x.shape[1]}"
            )

        train_idx, val_idx = self._train_val_indices(len(x))
        self.x_scaler_ = StandardScaler().fit(x[train_idx])
        x_scaled = self.x_scaler_.transform(x).astype(np.float32)
        self.y_mean_ = float(y[train_idx].mean())
        y_std = float(y[train_idx].std())
        self.y_std_ = y_std if y_std > 1e-8 else 1.0
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        self.model_ = self._make_net(
            **self._net_kwargs(),
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

    def _net_kwargs(self) -> dict:
        return {
            "metadata_dim": self.metadata_dim,
            "text_dim": self.text_dim,
            "image_dim": self.image_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
        }

    def _expected_dim(self) -> int:
        return self.metadata_dim + self.text_dim + self.image_dim

    def _feature_shape_label(self) -> str:
        return f"{self.metadata_dim} metadata + {self.text_dim} text + {self.image_dim} image"

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

    def _make_net(self, **kwargs):
        return _make_project_input_cross_attention_net(**kwargs)


class ProjectInputRecurrentFusionRegressor(ProjectInputCrossAttentionRegressor):
    """Project-input CTNN-style proxy with cross-attention followed by recurrent token fusion."""

    model_label = "project_input_recurrent_fusion"
    baseline_label = "C2-ProjectInputRecurrentFusion"

    def _make_net(self, **kwargs):
        return _make_project_input_recurrent_fusion_net(**kwargs)


class ProjectInputCTNNReconstructionRegressor(ProjectInputCrossAttentionRegressor):
    """Structure-complete project-input CTNN reconstruction.

    The original CTNN route combines text/poster transformer encoders,
    cross-modal attention, recurrent fusion, and metadata-related factors.
    This project-input version keeps anime synopsis/cover/banner/factor inputs
    while restoring those major modeling stages.
    """

    model_label = "project_input_ctnn_reconstruction"
    baseline_label = "C2-ProjectInputCTNNReconstruction"

    def __init__(
        self,
        metadata_dim: int = 151,
        text_dim: int = 768,
        image_dim: int = 4098,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        text_tokens: int = 4,
        image_feature_dim: int = 2048,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        super().__init__(
            metadata_dim=metadata_dim,
            text_dim=text_dim,
            image_dim=image_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            random_state=random_state,
            device=device,
            torch_num_threads=torch_num_threads,
            verbose=verbose,
        )
        self.text_tokens = text_tokens
        self.image_feature_dim = image_feature_dim

    def _net_kwargs(self) -> dict:
        kwargs = super()._net_kwargs()
        kwargs.update(
            {
                "text_tokens": self.text_tokens,
                "image_feature_dim": self.image_feature_dim,
            }
        )
        return kwargs

    def _make_net(self, **kwargs):
        return _make_project_input_ctnn_reconstruction_net(**kwargs)


class ProjectInputCTNNDualVisualReconstructionRegressor(ProjectInputCTNNReconstructionRegressor):
    """CTNN reconstruction with ResNet-style and transformer-style visual tokens.

    The CTNN paper extracts movie-poster features from ResNet50 and ViT. Under
    the project input contract, this diagnostic row keeps cover/banner images
    but combines project image embeddings with ResNet-50 cover/banner features.
    """

    model_label = "project_input_ctnn_dual_visual_reconstruction"
    baseline_label = "C2-ProjectInputCTNNDualVisualReconstruction"

    def __init__(
        self,
        metadata_dim: int = 151,
        text_dim: int = 768,
        swin_image_dim: int = 1024,
        resnet_image_dim: int = 4098,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        text_tokens: int = 4,
        image_feature_dim: int = 2048,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        super().__init__(
            metadata_dim=metadata_dim,
            text_dim=text_dim,
            image_dim=swin_image_dim + resnet_image_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            text_tokens=text_tokens,
            image_feature_dim=image_feature_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            random_state=random_state,
            device=device,
            torch_num_threads=torch_num_threads,
            verbose=verbose,
        )
        self.swin_image_dim = swin_image_dim
        self.resnet_image_dim = resnet_image_dim

    def _feature_shape_label(self) -> str:
        return (
            f"{self.metadata_dim} metadata + {self.text_dim} GPT-2 text + "
            f"{self.swin_image_dim} project image + {self.resnet_image_dim} ResNet-50 image"
        )

    def _net_kwargs(self) -> dict:
        kwargs = super()._net_kwargs()
        kwargs.update(
            {
                "swin_image_dim": self.swin_image_dim,
                "resnet_image_dim": self.resnet_image_dim,
            }
        )
        return kwargs

    def _make_net(self, **kwargs):
        return _make_project_input_ctnn_dual_visual_reconstruction_net(**kwargs)


class ProjectInputSKAPPGraphProxyRegressor(ProjectInputCrossAttentionRegressor):
    """Project-input SKAPP-style proxy with RRCP-selected retrieved graph context."""

    model_label = "project_input_skapp_graph_proxy"
    baseline_label = "C3-ProjectInputSKAPPGraphProxy"

    def __init__(
        self,
        metadata_dim: int = 151,
        rag_dim: int = 14096,
        rag_aggregate_dim: int = 16,
        text_dim: int = 384,
        image_dim: int = 1024,
        top_k: int = 10,
        label_dim: int = 2,
        d_model: int = 128,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        rrcp_threshold: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 80,
        patience: int = 10,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        super().__init__(
            metadata_dim=metadata_dim,
            text_dim=text_dim,
            image_dim=image_dim,
            d_model=d_model,
            nhead=1,
            num_layers=1,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            random_state=random_state,
            device=device,
            torch_num_threads=torch_num_threads,
            verbose=verbose,
        )
        self.rag_dim = rag_dim
        self.rag_aggregate_dim = rag_aggregate_dim
        self.top_k = top_k
        self.label_dim = label_dim
        self.rrcp_threshold = rrcp_threshold

    def _expected_dim(self) -> int:
        return self.metadata_dim + self.rag_dim + self.text_dim + self.image_dim

    def _feature_shape_label(self) -> str:
        return (
            f"{self.metadata_dim} metadata + {self.rag_dim} SKAPP graph/RAG "
            f"+ {self.text_dim} text + {self.image_dim} image"
        )

    def _net_kwargs(self) -> dict:
        return {
            "metadata_dim": self.metadata_dim,
            "rag_dim": self.rag_dim,
            "rag_aggregate_dim": self.rag_aggregate_dim,
            "text_dim": self.text_dim,
            "image_dim": self.image_dim,
            "top_k": self.top_k,
            "label_dim": self.label_dim,
            "d_model": self.d_model,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "rrcp_threshold": self.rrcp_threshold,
        }

    def _make_net(self, **kwargs):
        return _make_project_input_skapp_graph_proxy_net(**kwargs)


class BranchMLPRegressor:
    """Small sklearn-style branch MLP for metadata/text/image fusion."""

    def __init__(
        self,
        metadata_dim: int = 151,
        text_dim: int = 384,
        image_dim: int = 1024,
        branch_dim: int = 128,
        fusion_hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        self.metadata_dim = metadata_dim
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.branch_dim = branch_dim
        self.fusion_hidden_dims = tuple(fusion_hidden_dims)
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
            raise ImportError("torch is not installed; install it or disable C1-Armenta-ProxyBranchMLP") from exc

        if self.torch_num_threads:
            torch.set_num_threads(int(self.torch_num_threads))
        self._torch = torch
        self._device = torch.device(self._resolve_device(torch))
        self._set_seed(torch)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        expected_dim = self.metadata_dim + self.text_dim + self.image_dim
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"branch_mlp expects {expected_dim} features "
                f"({self.metadata_dim} metadata + {self.text_dim} text + {self.image_dim} image), "
                f"got {x.shape[1]}"
            )

        train_idx, val_idx = self._train_val_indices(len(x))
        self.x_scaler_ = StandardScaler().fit(x[train_idx])
        x_scaled = self.x_scaler_.transform(x).astype(np.float32)
        self.y_mean_ = float(y[train_idx].mean())
        y_std = float(y[train_idx].std())
        self.y_std_ = y_std if y_std > 1e-8 else 1.0
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        self.model_ = _make_branch_mlp_net(
            metadata_dim=self.metadata_dim,
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            branch_dim=self.branch_dim,
            fusion_hidden_dims=self.fusion_hidden_dims,
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


class ArmentaProjectInputMLPRegressor:
    """Armenta-shaped project-input proxy with a context MLP and Big MLP."""

    def __init__(
        self,
        metadata_dim: int = 151,
        text_dim: int = 384,
        image_dim: int = 1024,
        branch_dim: int = 768,
        big_mlp_hidden_dims: tuple[int, ...] = (768, 384, 192, 96, 48, 24, 12, 6),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        self.metadata_dim = metadata_dim
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.branch_dim = branch_dim
        self.big_mlp_hidden_dims = tuple(big_mlp_hidden_dims)
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
            raise ImportError("torch is not installed; install it or disable C1-Armenta-ProjectInputProxy") from exc

        if self.torch_num_threads:
            torch.set_num_threads(int(self.torch_num_threads))
        self._torch = torch
        self._device = torch.device(self._resolve_device(torch))
        self._set_seed(torch)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        expected_dim = self.metadata_dim + self.text_dim + self.image_dim
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"armenta_project_input_mlp expects {expected_dim} features "
                f"({self.metadata_dim} metadata + {self.text_dim} text + {self.image_dim} image), "
                f"got {x.shape[1]}"
            )

        train_idx, val_idx = self._train_val_indices(len(x))
        self.x_scaler_ = StandardScaler().fit(x[train_idx])
        x_scaled = self.x_scaler_.transform(x).astype(np.float32)
        self.y_mean_ = float(y[train_idx].mean())
        y_std = float(y[train_idx].std())
        self.y_std_ = y_std if y_std > 1e-8 else 1.0
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        self.model_ = _make_armenta_project_input_mlp_net(
            metadata_dim=self.metadata_dim,
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            branch_dim=self.branch_dim,
            big_mlp_hidden_dims=self.big_mlp_hidden_dims,
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


class ArmentaFigure2ReconstructionRegressor:
    """Armenta Figure 2 side reconstruction with synopsis, character text, and portrait features."""

    def __init__(
        self,
        synopsis_dim: int = 768,
        character_text_dim: int = 768,
        portrait_dim: int = 49,
        branch_dim: int = 768,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "cpu",
        torch_num_threads: int | None = 1,
        verbose: bool = False,
    ):
        self.synopsis_dim = synopsis_dim
        self.character_text_dim = character_text_dim
        self.portrait_dim = portrait_dim
        self.branch_dim = branch_dim
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
            raise ImportError("torch is not installed; install it or disable C1-Armenta-Figure2Reconstruction") from exc

        if self.torch_num_threads:
            torch.set_num_threads(int(self.torch_num_threads))
        self._torch = torch
        self._device = torch.device(self._resolve_device(torch))
        self._set_seed(torch)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        expected_dim = self.synopsis_dim + self.character_text_dim + self.portrait_dim
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"armenta_figure2_reconstruction expects {expected_dim} features "
                f"({self.synopsis_dim} synopsis + {self.character_text_dim} character text "
                f"+ {self.portrait_dim} portrait), got {x.shape[1]}"
            )

        train_idx, val_idx = self._train_val_indices(len(x))
        self.x_scaler_ = StandardScaler().fit(x[train_idx])
        x_scaled = self.x_scaler_.transform(x).astype(np.float32)
        self.y_mean_ = float(y[train_idx].mean())
        y_std = float(y[train_idx].std())
        self.y_std_ = y_std if y_std > 1e-8 else 1.0
        y_scaled = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        self.model_ = _make_armenta_figure2_reconstruction_net(
            synopsis_dim=self.synopsis_dim,
            character_text_dim=self.character_text_dim,
            portrait_dim=self.portrait_dim,
            branch_dim=self.branch_dim,
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


def _make_branch_mlp_net(
    metadata_dim: int,
    text_dim: int,
    image_dim: int,
    branch_dim: int,
    fusion_hidden_dims: tuple[int, ...],
    dropout: float,
):
    import torch
    from torch import nn

    def branch(in_dim: int):
        mid_dim = max(branch_dim * 2, branch_dim)
        return nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.metadata_branch = branch(metadata_dim)
            self.text_branch = branch(text_dim)
            self.image_branch = branch(image_dim)

            layers = []
            in_dim = branch_dim * 3
            for hidden_dim in fusion_hidden_dims:
                layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self.fusion = nn.Sequential(*layers)

        def forward(self, x):
            metadata = x[:, :metadata_dim]
            text_start = metadata_dim
            image_start = metadata_dim + text_dim
            text = x[:, text_start:image_start]
            image = x[:, image_start : image_start + image_dim]
            fused = [
                self.metadata_branch(metadata),
                self.text_branch(text),
                self.image_branch(image),
            ]
            return self.fusion(torch.cat(fused, dim=1))

    return Net()


def _make_armenta_project_input_mlp_net(
    metadata_dim: int,
    text_dim: int,
    image_dim: int,
    branch_dim: int,
    big_mlp_hidden_dims: tuple[int, ...],
    dropout: float,
):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.synopsis_branch = nn.Sequential(
                nn.Linear(text_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.project_context_mlp = nn.Sequential(
                nn.Linear(metadata_dim + image_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(branch_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
            )

            layers = []
            in_dim = branch_dim * 2
            for hidden_dim in big_mlp_hidden_dims:
                layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self.big_mlp = nn.Sequential(*layers)

        def forward(self, x):
            metadata = x[:, :metadata_dim]
            text_start = metadata_dim
            image_start = metadata_dim + text_dim
            text = x[:, text_start:image_start]
            image = x[:, image_start : image_start + image_dim]
            synopsis = self.synopsis_branch(text)
            context = self.project_context_mlp(torch.cat([metadata, image], dim=1))
            return self.big_mlp(torch.cat([synopsis, context], dim=1))

    return Net()


def _make_armenta_figure2_reconstruction_net(
    synopsis_dim: int,
    character_text_dim: int,
    portrait_dim: int,
    branch_dim: int,
    dropout: float,
):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.character_mlp = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(character_text_dim + portrait_dim, branch_dim, bias=True),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(branch_dim, branch_dim, bias=True),
            )
            self.synopsis_branch = nn.Identity()
            self.big_mlp = nn.Sequential(
                nn.Linear(branch_dim * 2, 768),
                nn.Tanh(),
                nn.Linear(768, 384),
                nn.Tanh(),
                nn.Linear(384, 192),
                nn.Tanh(),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 12),
                nn.ReLU(),
                nn.Linear(12, 6),
                nn.ReLU(),
                nn.Linear(6, 1),
            )

        def forward(self, x):
            syn = x[:, :synopsis_dim]
            char_text_start = synopsis_dim
            portrait_start = synopsis_dim + character_text_dim
            char_text = x[:, char_text_start:portrait_start]
            portrait = x[:, portrait_start : portrait_start + portrait_dim]
            char = self.character_mlp(torch.cat([char_text, portrait], dim=1))
            return self.big_mlp(torch.cat([self.synopsis_branch(syn), char], dim=1))

    return Net()


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


def _make_project_input_cross_attention_net(
    metadata_dim: int,
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

    class CrossAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_to_image = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.image_to_text = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.text_norm = nn.LayerNorm(d_model)
            self.image_norm = nn.LayerNorm(d_model)
            self.text_ffn = _feedforward_block(d_model, dim_feedforward, dropout)
            self.image_ffn = _feedforward_block(d_model, dim_feedforward, dropout)

        def forward(self, text_token, image_token):
            text_delta, _ = self.text_to_image(text_token, image_token, image_token, need_weights=False)
            image_delta, _ = self.image_to_text(image_token, text_token, text_token, need_weights=False)
            text_token = self.text_norm(text_token + text_delta)
            image_token = self.image_norm(image_token + image_delta)
            text_token = text_token + self.text_ffn(text_token)
            image_token = image_token + self.image_ffn(image_token)
            return text_token, image_token

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.metadata_proj = nn.Sequential(
                nn.Linear(metadata_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.text_proj = nn.Sequential(nn.Linear(text_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.image_proj = nn.Sequential(nn.Linear(image_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.cross_blocks = nn.ModuleList([CrossAttentionBlock() for _ in range(num_layers)])
            self.gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 3),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            metadata = x[:, :metadata_dim]
            text_start = metadata_dim
            image_start = metadata_dim + text_dim
            text = x[:, text_start:image_start]
            image = x[:, image_start : image_start + image_dim]

            metadata_token = self.metadata_proj(metadata).unsqueeze(1)
            text_token = self.text_proj(text).unsqueeze(1)
            image_token = self.image_proj(image).unsqueeze(1)
            for block in self.cross_blocks:
                text_token, image_token = block(text_token, image_token)

            tokens = torch.cat([text_token, image_token, metadata_token], dim=1)
            weights = torch.softmax(self.gate(metadata_token.squeeze(1)), dim=1).unsqueeze(-1)
            fused = (tokens * weights).sum(dim=1)
            return self.head(fused)

    return Net()


def _make_project_input_recurrent_fusion_net(
    metadata_dim: int,
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

    class CrossAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_to_image = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.image_to_text = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.text_norm = nn.LayerNorm(d_model)
            self.image_norm = nn.LayerNorm(d_model)
            self.text_ffn = _feedforward_block(d_model, dim_feedforward, dropout)
            self.image_ffn = _feedforward_block(d_model, dim_feedforward, dropout)

        def forward(self, text_token, image_token):
            text_delta, _ = self.text_to_image(text_token, image_token, image_token, need_weights=False)
            image_delta, _ = self.image_to_text(image_token, text_token, text_token, need_weights=False)
            text_token = self.text_norm(text_token + text_delta)
            image_token = self.image_norm(image_token + image_delta)
            text_token = text_token + self.text_ffn(text_token)
            image_token = image_token + self.image_ffn(image_token)
            return text_token, image_token

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.metadata_proj = nn.Sequential(
                nn.Linear(metadata_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.text_proj = nn.Sequential(nn.Linear(text_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.image_proj = nn.Sequential(nn.Linear(image_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            self.cross_blocks = nn.ModuleList([CrossAttentionBlock() for _ in range(num_layers)])
            self.recurrent_fusion = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                dropout=0.0,
                batch_first=True,
            )
            self.metadata_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            metadata = x[:, :metadata_dim]
            text_start = metadata_dim
            image_start = metadata_dim + text_dim
            text = x[:, text_start:image_start]
            image = x[:, image_start : image_start + image_dim]

            metadata_token = self.metadata_proj(metadata).unsqueeze(1)
            text_token = self.text_proj(text).unsqueeze(1)
            image_token = self.image_proj(image).unsqueeze(1)
            for block in self.cross_blocks:
                text_token, image_token = block(text_token, image_token)

            token_sequence = torch.cat([text_token, image_token, metadata_token], dim=1)
            recurrent_outputs, recurrent_state = self.recurrent_fusion(token_sequence)
            recurrent_summary = recurrent_state[-1]
            metadata_gate = self.metadata_gate(metadata_token.squeeze(1))
            gated_context = recurrent_outputs.mean(dim=1) * metadata_gate
            return self.head(torch.cat([recurrent_summary, gated_context], dim=1))

    return Net()


def _make_project_input_ctnn_reconstruction_net(
    metadata_dim: int,
    text_dim: int,
    image_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    text_tokens: int,
    image_feature_dim: int,
):
    import torch
    from torch import nn

    if text_dim % text_tokens != 0:
        raise ValueError(f"text_dim={text_dim} must be divisible by text_tokens={text_tokens}")
    text_token_dim = text_dim // text_tokens

    class CrossAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_to_image = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.image_to_text = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.text_norm = nn.LayerNorm(d_model)
            self.image_norm = nn.LayerNorm(d_model)
            self.text_ffn = _feedforward_block(d_model, dim_feedforward, dropout)
            self.image_ffn = _feedforward_block(d_model, dim_feedforward, dropout)

        def forward(self, text_tokens, image_tokens):
            text_delta, _ = self.text_to_image(text_tokens, image_tokens, image_tokens, need_weights=False)
            image_delta, _ = self.image_to_text(image_tokens, text_tokens, text_tokens, need_weights=False)
            text_tokens = self.text_norm(text_tokens + text_delta)
            image_tokens = self.image_norm(image_tokens + image_delta)
            text_tokens = text_tokens + self.text_ffn(text_tokens)
            image_tokens = image_tokens + self.image_ffn(image_tokens)
            return text_tokens, image_tokens

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_token_proj = nn.Sequential(
                nn.Linear(text_token_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.image_token_proj = nn.Sequential(
                nn.Linear(image_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.text_pos = nn.Parameter(torch.zeros(1, text_tokens, d_model))
            self.image_pos = nn.Parameter(torch.zeros(1, 2, d_model))

            text_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            image_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=num_layers)
            self.image_encoder = nn.TransformerEncoder(image_layer, num_layers=num_layers)
            self.cross_blocks = nn.ModuleList([CrossAttentionBlock() for _ in range(num_layers)])

            self.metadata_factor = nn.Sequential(
                nn.Linear(metadata_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.factor_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 3),
            )
            self.recurrent_fusion = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            metadata = x[:, :metadata_dim]
            text_start = metadata_dim
            image_start = metadata_dim + text_dim
            text = x[:, text_start:image_start]
            image = x[:, image_start : image_start + image_dim]

            text_seq = text.reshape(text.shape[0], text_tokens, text_token_dim)
            text_seq = self.text_token_proj(text_seq) + self.text_pos

            if image.shape[1] >= image_feature_dim * 2:
                cover = image[:, :image_feature_dim]
                banner = image[:, image_feature_dim : image_feature_dim * 2]
                availability = image[:, image_feature_dim * 2 : image_feature_dim * 2 + 2]
                if availability.shape[1] == 2:
                    cover = cover * availability[:, 0:1]
                    banner = banner * availability[:, 1:2]
                image_seq = torch.stack([cover, banner], dim=1)
            else:
                first, second = torch.chunk(image, chunks=2, dim=1)
                if first.shape[1] != image_feature_dim:
                    first = torch.nn.functional.pad(first, (0, image_feature_dim - first.shape[1]))
                if second.shape[1] != image_feature_dim:
                    second = torch.nn.functional.pad(second, (0, image_feature_dim - second.shape[1]))
                image_seq = torch.stack([first, second], dim=1)
            image_seq = self.image_token_proj(image_seq) + self.image_pos

            text_seq = self.text_encoder(text_seq)
            image_seq = self.image_encoder(image_seq)
            for block in self.cross_blocks:
                text_seq, image_seq = block(text_seq, image_seq)

            text_summary = text_seq.mean(dim=1)
            image_summary = image_seq.mean(dim=1)
            factor = self.metadata_factor(metadata)
            factors = torch.stack([text_summary, image_summary, factor], dim=1)
            gate = torch.softmax(self.factor_gate(factor), dim=1).unsqueeze(-1)
            gated_summary = (factors * gate).sum(dim=1)
            _, recurrent_state = self.recurrent_fusion(factors)
            recurrent_summary = recurrent_state[-1]
            return self.head(torch.cat([gated_summary, recurrent_summary], dim=1))

    return Net()


def _make_project_input_ctnn_dual_visual_reconstruction_net(
    metadata_dim: int,
    text_dim: int,
    image_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    text_tokens: int,
    image_feature_dim: int,
    swin_image_dim: int,
    resnet_image_dim: int,
):
    import torch
    from torch import nn

    if image_dim != swin_image_dim + resnet_image_dim:
        raise ValueError(
            f"image_dim={image_dim} must equal swin_image_dim + resnet_image_dim "
            f"({swin_image_dim + resnet_image_dim})"
        )
    if text_dim % text_tokens != 0:
        raise ValueError(f"text_dim={text_dim} must be divisible by text_tokens={text_tokens}")
    if swin_image_dim % 2 != 0:
        raise ValueError(f"swin_image_dim={swin_image_dim} must split into cover/banner halves")
    text_token_dim = text_dim // text_tokens
    swin_token_dim = swin_image_dim // 2

    class CrossAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_to_image = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.image_to_text = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.text_norm = nn.LayerNorm(d_model)
            self.image_norm = nn.LayerNorm(d_model)
            self.text_ffn = _feedforward_block(d_model, dim_feedforward, dropout)
            self.image_ffn = _feedforward_block(d_model, dim_feedforward, dropout)

        def forward(self, text_tokens, image_tokens):
            text_delta, _ = self.text_to_image(text_tokens, image_tokens, image_tokens, need_weights=False)
            image_delta, _ = self.image_to_text(image_tokens, text_tokens, text_tokens, need_weights=False)
            text_tokens = self.text_norm(text_tokens + text_delta)
            image_tokens = self.image_norm(image_tokens + image_delta)
            text_tokens = text_tokens + self.text_ffn(text_tokens)
            image_tokens = image_tokens + self.image_ffn(image_tokens)
            return text_tokens, image_tokens

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_token_proj = nn.Sequential(
                nn.Linear(text_token_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.resnet_token_proj = nn.Sequential(
                nn.Linear(image_feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.swin_token_proj = nn.Sequential(
                nn.Linear(swin_token_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.text_pos = nn.Parameter(torch.zeros(1, text_tokens, d_model))
            self.image_pos = nn.Parameter(torch.zeros(1, 4, d_model))

            text_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            image_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=num_layers)
            self.image_encoder = nn.TransformerEncoder(image_layer, num_layers=num_layers)
            self.cross_blocks = nn.ModuleList([CrossAttentionBlock() for _ in range(num_layers)])

            self.metadata_factor = nn.Sequential(
                nn.Linear(metadata_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.factor_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 3),
            )
            self.recurrent_fusion = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            metadata = x[:, :metadata_dim]
            text_start = metadata_dim
            swin_start = text_start + text_dim
            resnet_start = swin_start + swin_image_dim

            text = x[:, text_start:swin_start]
            swin = x[:, swin_start:resnet_start]
            resnet = x[:, resnet_start : resnet_start + resnet_image_dim]

            text_seq = text.reshape(text.shape[0], text_tokens, text_token_dim)
            text_seq = self.text_token_proj(text_seq) + self.text_pos

            swin_cover, swin_banner = torch.chunk(swin, chunks=2, dim=1)
            swin_seq = torch.stack([swin_cover, swin_banner], dim=1)
            swin_seq = self.swin_token_proj(swin_seq)

            if resnet.shape[1] >= image_feature_dim * 2:
                cover = resnet[:, :image_feature_dim]
                banner = resnet[:, image_feature_dim : image_feature_dim * 2]
                availability = resnet[:, image_feature_dim * 2 : image_feature_dim * 2 + 2]
                if availability.shape[1] == 2:
                    cover = cover * availability[:, 0:1]
                    banner = banner * availability[:, 1:2]
            else:
                cover, banner = torch.chunk(resnet, chunks=2, dim=1)
                if cover.shape[1] != image_feature_dim:
                    cover = torch.nn.functional.pad(cover, (0, image_feature_dim - cover.shape[1]))
                if banner.shape[1] != image_feature_dim:
                    banner = torch.nn.functional.pad(banner, (0, image_feature_dim - banner.shape[1]))
            resnet_seq = torch.stack([cover, banner], dim=1)
            resnet_seq = self.resnet_token_proj(resnet_seq)

            image_seq = torch.cat([resnet_seq, swin_seq], dim=1) + self.image_pos
            text_seq = self.text_encoder(text_seq)
            image_seq = self.image_encoder(image_seq)
            for block in self.cross_blocks:
                text_seq, image_seq = block(text_seq, image_seq)

            text_summary = text_seq.mean(dim=1)
            image_summary = image_seq.mean(dim=1)
            factor = self.metadata_factor(metadata)
            factors = torch.stack([text_summary, image_summary, factor], dim=1)
            gate = torch.softmax(self.factor_gate(factor), dim=1).unsqueeze(-1)
            gated_summary = (factors * gate).sum(dim=1)
            _, recurrent_state = self.recurrent_fusion(factors)
            recurrent_summary = recurrent_state[-1]
            return self.head(torch.cat([gated_summary, recurrent_summary], dim=1))

    return Net()


def _make_project_input_skapp_graph_proxy_net(
    metadata_dim: int,
    rag_dim: int,
    rag_aggregate_dim: int,
    text_dim: int,
    image_dim: int,
    top_k: int,
    label_dim: int,
    d_model: int,
    dim_feedforward: int,
    dropout: float,
    rrcp_threshold: float,
):
    import torch
    from torch import nn

    retrieved_dim = text_dim + image_dim + label_dim + 1

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.metadata_proj = nn.Sequential(
                nn.Linear(metadata_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.image_proj = nn.Sequential(
                nn.Linear(image_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.retrieved_proj = nn.Sequential(
                nn.Linear(retrieved_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.graph_query = nn.Linear(d_model, d_model, bias=False)
            self.graph_key = nn.Linear(d_model, d_model, bias=False)
            self.graph_value = nn.Linear(d_model, d_model, bias=False)
            self.rrcp_attention = nn.Sequential(
                nn.Linear(d_model + 1, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 1),
            )
            self.aggregate_proj = nn.Sequential(
                nn.Linear(rag_aggregate_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 4),
                nn.Linear(d_model * 4, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 1),
            )

        def forward(self, x):
            metadata_end = metadata_dim
            rag_end = metadata_end + rag_dim
            text_end = rag_end + text_dim

            metadata = x[:, :metadata_end]
            rag = x[:, metadata_end:rag_end]
            query_text = x[:, rag_end:text_end]
            query_image = x[:, text_end : text_end + image_dim]

            aggregate = rag[:, :rag_aggregate_dim]
            offset = rag_aggregate_dim
            mask = rag[:, offset : offset + top_k]
            offset += top_k
            rrcp = rag[:, offset : offset + top_k]
            offset += top_k
            labels = rag[:, offset : offset + top_k * label_dim].reshape(-1, top_k, label_dim)
            offset += top_k * label_dim
            retrieved_text = rag[:, offset : offset + top_k * text_dim].reshape(-1, top_k, text_dim)
            offset += top_k * text_dim
            retrieved_image = rag[:, offset : offset + top_k * image_dim].reshape(-1, top_k, image_dim)

            query_token = (
                self.metadata_proj(metadata)
                + self.text_proj(query_text)
                + self.image_proj(query_image)
            ) / 3.0
            retrieved_raw = torch.cat(
                [retrieved_text, retrieved_image, labels, rrcp.unsqueeze(-1)],
                dim=-1,
            )
            retrieved_tokens = self.retrieved_proj(retrieved_raw)

            selected_mask = (mask > 0.0).float()
            if rrcp_threshold > 0.0:
                selected_mask = selected_mask * (rrcp > rrcp_threshold).float()
            fallback_mask = (selected_mask.sum(dim=1, keepdim=True) <= 0).float()
            selected_mask = selected_mask + fallback_mask * (mask > 0.0).float()

            graph_tokens = torch.cat([query_token.unsqueeze(1), retrieved_tokens], dim=1)
            graph_mask = torch.cat(
                [torch.ones_like(selected_mask[:, :1]), selected_mask],
                dim=1,
            )
            q = self.graph_query(graph_tokens)
            k = self.graph_key(graph_tokens)
            v = self.graph_value(graph_tokens)
            adjacency = torch.matmul(q, k.transpose(1, 2)) / float(d_model) ** 0.5
            adjacency = adjacency.masked_fill(graph_mask.unsqueeze(1) <= 0, -1e4)
            graph_context = torch.softmax(adjacency, dim=-1).matmul(v)[:, 0, :]

            attention_input = torch.cat([retrieved_tokens, rrcp.unsqueeze(-1)], dim=-1)
            attention_logits = self.rrcp_attention(attention_input).squeeze(-1)
            attention_logits = attention_logits.masked_fill(selected_mask <= 0, -1e4)
            attention = torch.softmax(attention_logits, dim=-1).unsqueeze(-1)
            retrieved_context = (attention * retrieved_tokens).sum(dim=1)
            aggregate_context = self.aggregate_proj(aggregate)
            return self.head(
                torch.cat([query_token, graph_context, retrieved_context, aggregate_context], dim=1)
            )

    return Net()


def _feedforward_block(d_model: int, dim_feedforward: int, dropout: float):
    from torch import nn

    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, dim_feedforward),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
        nn.Dropout(dropout),
    )
