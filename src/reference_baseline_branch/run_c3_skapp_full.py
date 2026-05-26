from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.experiment_common.metrics import compute_metrics
from src.reference_baseline_branch.build_c3_rag_features import OfflineRagFeatureBuilder


TARGET_SPECS = {
    "popularity": {"retrieved_label": 0, "target_transform": "log1p"},
    "meanScore": {"retrieved_label": 1, "target_transform": "identity"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a structure-complete project-input SKAPP reconstruction."
    )
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
        help="Reference baseline config YAML.",
    )
    parser.add_argument("--target", choices=tuple(TARGET_SPECS), default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--single-dropout", type=float, default=None)
    parser.add_argument("--threshold-of-rrcp", type=float, default=0.0)
    parser.add_argument("--variant-label", default="")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--force-rebuild-dataset", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    run_dir = _resolve_run_dir(config, args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(config["data"].get("skapp_full_dataset_dir", ".exp/baseline/skapp_full/dataset"))
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.force_rebuild_dataset or not _dataset_exists(dataset_dir):
        _build_tensor_dataset(config, dataset_dir, top_k=args.top_k)

    targets = [args.target] if args.target else list(TARGET_SPECS)
    rows = []
    for target in targets:
        print(f"[skapp-full] target={target}")
        row = _run_target(config, dataset_dir, run_dir, target, args)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "baseline_results.csv", index=False)
    with open(run_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    _write_summary(table, run_dir / "baseline_summary.md")
    print(f"[done] results saved to {run_dir}")


def _build_tensor_dataset(config: dict, dataset_dir: Path, top_k: int) -> None:
    print(f"[skapp-full] build tensor dataset top_k={top_k}")
    builder = OfflineRagFeatureBuilder(config, top_k=top_k)
    splits = config["data"].get("splits", ["train", "val", "test"])
    for split in splits:
        df = builder.meta[split].reset_index(drop=True)
        rows = [_tensor_row(builder, row, split, top_k) for _, row in df.iterrows()]
        ids = np.asarray([item["id"] for item in rows], dtype=np.int64)
        query_text = np.stack([item["query_text"] for item in rows]).astype(np.float32)
        query_image = np.stack([item["query_image"] for item in rows]).astype(np.float32)
        retrieved_text = np.stack([item["retrieved_text"] for item in rows]).astype(np.float32)
        retrieved_image = np.stack([item["retrieved_image"] for item in rows]).astype(np.float32)
        retrieved_labels = np.stack([item["retrieved_labels"] for item in rows]).astype(np.float32)
        retrieved_mask = np.stack([item["retrieved_mask"] for item in rows]).astype(np.float32)
        y_popularity = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        y_mean_score = pd.to_numeric(df["meanScore"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        out_path = dataset_dir / f"{split}.npz"
        np.savez_compressed(
            out_path,
            ids=ids,
            query_text=query_text,
            query_image=query_image,
            retrieved_text=retrieved_text,
            retrieved_image=retrieved_image,
            retrieved_labels=retrieved_labels,
            retrieved_mask=retrieved_mask,
            popularity=y_popularity,
            meanScore=y_mean_score,
        )
        valid = float(retrieved_mask.mean()) if retrieved_mask.size else 0.0
        print(f"  [{split}] {len(ids)} rows -> {out_path}  mask_mean={valid:.4f}")


def _tensor_row(
    builder: OfflineRagFeatureBuilder,
    row: pd.Series,
    split: str,
    top_k: int,
) -> dict:
    anime_id = int(row[builder.id_col])
    candidates = builder._retrieve(row, split, "hybrid")
    if not candidates:
        candidates = builder._retrieve(row, split, "sparse")
    candidates = candidates[:top_k]

    query_text = _query_vector(builder.text_emb.get(split), anime_id, builder.text_dim)
    query_image = _query_vector(builder.image_emb.get(split), anime_id, builder.image_dim)
    retrieved_text = np.zeros((top_k, builder.text_dim), dtype=np.float32)
    retrieved_image = np.zeros((top_k, builder.image_dim), dtype=np.float32)
    retrieved_labels = np.zeros((top_k, 2), dtype=np.float32)
    retrieved_mask = np.zeros(top_k, dtype=np.float32)

    for pos, (idx, _) in enumerate(candidates):
        candidate = builder.train_df.iloc[idx]
        retrieved_mask[pos] = 1.0
        if builder.train_text_matrix is not None:
            retrieved_text[pos] = builder.train_text_matrix[idx]
        if builder.train_image_matrix is not None:
            retrieved_image[pos] = builder.train_image_matrix[idx]
        retrieved_labels[pos, 0] = math.log1p(max(_safe_float(candidate.get("popularity")), 0.0))
        retrieved_labels[pos, 1] = _safe_float(candidate.get("meanScore")) / 100.0

    return {
        "id": anime_id,
        "query_text": query_text,
        "query_image": query_image,
        "retrieved_text": retrieved_text,
        "retrieved_image": retrieved_image,
        "retrieved_labels": retrieved_labels,
        "retrieved_mask": retrieved_mask,
    }


def _query_vector(emb_map: dict | None, anime_id: int, dim: int) -> np.ndarray:
    if emb_map is None:
        return np.zeros(dim, dtype=np.float32)
    vec = emb_map.get(anime_id)
    if vec is None:
        return np.zeros(dim, dtype=np.float32)
    return np.asarray(vec, dtype=np.float32)


def _run_target(
    config: dict,
    dataset_dir: Path,
    run_dir: Path,
    target: str,
    args: argparse.Namespace,
) -> dict:
    import torch
    from torch import nn

    if args.torch_num_threads:
        torch.set_num_threads(int(args.torch_num_threads))
    _set_seed(torch, 42)
    device = _resolve_device(torch, args.device)
    data = {split: _load_split(dataset_dir / f"{split}.npz", target) for split in ("train", "val", "test")}
    y_mean = float(data["train"]["y_model"].mean())
    y_std = float(data["train"]["y_model"].std())
    if y_std <= 1e-8:
        y_std = 1.0
    for split_data in data.values():
        split_data["y_scaled"] = ((split_data["y_model"] - y_mean) / y_std).astype(np.float32)
        label = split_data["retrieved_label"]
        split_data["retrieved_label_scaled"] = ((label - y_mean) / y_std).astype(np.float32)

    dims = {
        "text_dim": int(data["train"]["query_text"].shape[1]),
        "image_dim": int(data["train"]["query_image"].shape[1]),
        "top_k": int(data["train"]["retrieved_text"].shape[1]),
        "d_model": int(args.d_model),
    }
    single_dropout = args.dropout if args.single_dropout is None else args.single_dropout

    all_model = _train_model(
        torch= torch,
        nn=nn,
        model=_SKAPPAllItemsModel(**dims, dropout=float(args.dropout)),
        train=data["train"],
        val=data["val"],
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        mode="all",
    )
    single_model = _train_model(
        torch=torch,
        nn=nn,
        model=_SKAPPSingleItemModel(**dims, dropout=float(single_dropout)),
        train=_make_disassembled_data(data["train"]),
        val=_make_disassembled_data(data["val"]),
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        mode="single",
    )

    diagnostics = {
        "target": target,
        "settings": {
            "top_k": int(args.top_k),
            "d_model": int(args.d_model),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "threshold_of_rrcp": float(args.threshold_of_rrcp),
            "dropout": float(args.dropout),
            "single_dropout": float(single_dropout),
            "variant_label": str(args.variant_label),
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "all_items_model": _collect_stage_diagnostics(
            torch=torch,
            model=all_model,
            data_by_split=data,
            splits=("train", "val", "test"),
            device=device,
            batch_size=args.batch_size,
            mode="all",
            target=target,
            y_mean=y_mean,
            y_std=y_std,
        ),
        "single_item_model": _collect_stage_diagnostics(
            torch=torch,
            model=single_model,
            data_by_split={
                "train": _make_disassembled_data(data["train"]),
                "val": _make_disassembled_data(data["val"]),
            },
            splits=("train", "val"),
            device=device,
            batch_size=args.batch_size,
            mode="single",
            target=target,
            y_mean=y_mean,
            y_std=y_std,
        ),
    }

    for split, split_data in data.items():
        split_data["rrcp_silver"] = _compute_rrcp_silver(
            torch=torch,
            all_model=all_model,
            single_model=single_model,
            data=split_data,
            device=device,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            run_dir / f"rrcp_silver_{target}_{split}.npz",
            ids=split_data["ids"],
            rrcp_silver=split_data["rrcp_silver"],
        )

    diagnostics["rrcp_silver"] = {
        split: _rrcp_summary(
            split_data["rrcp_silver"],
            split_data["retrieved_mask"],
            threshold=float(args.threshold_of_rrcp),
        )
        for split, split_data in data.items()
    }

    final_model = _train_model(
        torch=torch,
        nn=nn,
        model=_SKAPPFinalRRCPModel(
            **dims,
            threshold_of_rrcp=args.threshold_of_rrcp,
            dropout=float(args.dropout),
        ),
        train=data["train"],
        val=data["val"],
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        mode="final",
    )

    row = {
        "baseline_id": f"C3-ProjectInputSKAPPFull{args.variant_label}",
        "target": target,
        "feature_set": "skapp_style_tensor_dataset",
        "model": "project_input_skapp_full",
        "reference": "Xu et al. 2025 SKAPP",
        "reproduction_level": "structure_complete_project_input",
        "paper_supported_component": (
            "SKAPP uses all-items RRCP model, dissembled single-item model, "
            "RRCP_silver, threshold filtering, GraphLearner, and RRCP/CXMI attention."
        ),
        "project_adaptation_component": (
            "UGC social inputs are replaced by project anime query text/image and "
            "temporally valid historical anime retrieved text/image/label tensors."
        ),
        "claim_allowed": (
            "Project-input structure-complete SKAPP reconstruction with RRCP_silver "
            "and source-shaped graph/attention stages."
        ),
        "claim_not_allowed": (
            "Do not claim exact numerical SKAPP reproduction; domain, retrieval pool, "
            "feature encoders, and targets differ from the original social-media task."
        ),
        "status": "ok",
        "notes": (
            f"dropout={float(args.dropout):.3f}; "
            f"single_dropout={float(single_dropout):.3f}; "
            f"weight_decay={float(args.weight_decay):.6f}; "
            f"patience={int(args.patience)}"
        ),
        "n_train": int(len(data["train"]["ids"])),
        "n_features": int(
            dims["text_dim"]
            + dims["image_dim"]
            + dims["top_k"] * (dims["text_dim"] + dims["image_dim"] + 2)
        ),
    }

    pred_dir = run_dir / "predictions" / "C3-ProjectInputSKAPPFull" / target
    pred_dir.mkdir(parents=True, exist_ok=True)
    diagnostics["final_model"] = {}
    for split in ("val", "test"):
        pred_scaled = _predict(torch, final_model, data[split], device, args.batch_size, mode="final")
        pred_model = pred_scaled * y_std + y_mean
        pred_raw = _inverse_target(pred_model, target)
        diagnostics["final_model"][split] = _prediction_summary(
            y_raw=data[split]["y_raw"],
            y_model=data[split]["y_model"],
            pred_model=pred_model,
            target=target,
            train_y_model=data["train"]["y_model"],
        )
        metrics = compute_metrics(data[split]["y_raw"], pred_raw, target)
        for key, value in metrics.items():
            row[f"{split}_{key}"] = value
        row[f"n_{split}"] = int(len(data[split]["ids"]))
        pd.DataFrame(
            {
                "id": data[split]["ids"],
                "target": data[split]["y_raw"],
                "prediction": pred_raw,
            }
        ).to_csv(pred_dir / f"{split}_predictions.csv", index=False)

    with open(pred_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(_feature_names(dims), f, indent=2, ensure_ascii=False)
    with open(run_dir / f"c3_skapp_full_diagnostics_{target}.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(diagnostics), f, indent=2, ensure_ascii=False)
    return row


def _load_split(path: Path, target: str) -> dict:
    item = dict(np.load(path))
    spec = TARGET_SPECS[target]
    raw = item[target].astype(np.float32)
    if spec["target_transform"] == "log1p":
        y_model = np.log1p(np.clip(raw, 0, None)).astype(np.float32)
    else:
        y_model = raw.astype(np.float32)
    retrieved_label = item["retrieved_labels"][:, :, spec["retrieved_label"]].astype(np.float32)
    if target == "meanScore":
        retrieved_label = retrieved_label * 100.0
    item["y_raw"] = raw
    item["y_model"] = y_model
    item["retrieved_label"] = retrieved_label
    return item


def _make_disassembled_data(data: dict) -> dict:
    n_rows, top_k = data["retrieved_mask"].shape
    row_idx = np.repeat(np.arange(n_rows), top_k)
    item_idx = np.tile(np.arange(top_k), n_rows)
    mask = data["retrieved_mask"].reshape(-1) > 0
    row_idx = row_idx[mask]
    item_idx = item_idx[mask]
    return {
        "ids": data["ids"][row_idx],
        "query_text": data["query_text"][row_idx],
        "query_image": data["query_image"][row_idx],
        "retrieved_text_one": data["retrieved_text"][row_idx, item_idx],
        "retrieved_image_one": data["retrieved_image"][row_idx, item_idx],
        "retrieved_label_one": data["retrieved_label_scaled"][row_idx, item_idx],
        "y_scaled": data["y_scaled"][row_idx],
    }


def _train_model(
    torch,
    nn,
    model,
    train: dict,
    val: dict,
    device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    mode: str,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    stale = 0
    rng = np.random.default_rng(42)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train["y_scaled"]))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = _batch(train, order[start : start + batch_size], torch, device, mode)
            optimizer.zero_grad(set_to_none=True)
            pred = model(*batch[:-1]).reshape(-1)
            loss = loss_fn(pred, batch[-1])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(val["y_scaled"]), batch_size):
                idx = np.arange(start, min(start + batch_size, len(val["y_scaled"])))
                batch = _batch(val, idx, torch, device, mode)
                pred = model(*batch[:-1]).reshape(-1)
                val_losses.append(float(loss_fn(pred, batch[-1]).detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else np.inf
        train_loss = float(np.mean(losses)) if losses else np.nan
        print(f"  [{mode}] epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _batch(data: dict, idx: np.ndarray, torch, device, mode: str) -> Tuple:
    y = torch.from_numpy(data["y_scaled"][idx].astype(np.float32)).to(device)
    query_text = torch.from_numpy(data["query_text"][idx].astype(np.float32)).to(device)
    query_image = torch.from_numpy(data["query_image"][idx].astype(np.float32)).to(device)
    if mode == "single":
        retrieved_text = torch.from_numpy(data["retrieved_text_one"][idx].astype(np.float32)).to(device)
        retrieved_image = torch.from_numpy(data["retrieved_image_one"][idx].astype(np.float32)).to(device)
        retrieved_label = torch.from_numpy(data["retrieved_label_one"][idx].astype(np.float32)).to(device)
        return query_text, query_image, retrieved_text, retrieved_image, retrieved_label, y
    retrieved_text = torch.from_numpy(data["retrieved_text"][idx].astype(np.float32)).to(device)
    retrieved_image = torch.from_numpy(data["retrieved_image"][idx].astype(np.float32)).to(device)
    retrieved_label = torch.from_numpy(data["retrieved_label_scaled"][idx].astype(np.float32)).to(device)
    retrieved_mask = torch.from_numpy(data["retrieved_mask"][idx].astype(np.float32)).to(device)
    if mode == "final":
        rrcp = torch.from_numpy(data["rrcp_silver"][idx].astype(np.float32)).to(device)
        return query_text, query_image, retrieved_text, retrieved_image, retrieved_label, retrieved_mask, rrcp, y
    return query_text, query_image, retrieved_text, retrieved_image, retrieved_label, retrieved_mask, y


def _predict(torch, model, data: dict, device, batch_size: int, mode: str) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data["ids"]), batch_size):
            idx = np.arange(start, min(start + batch_size, len(data["ids"])))
            batch = _batch(data, idx, torch, device, mode)
            pred = model(*batch[:-1]).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def _compute_rrcp_silver(torch, all_model, single_model, data: dict, device, batch_size: int) -> np.ndarray:
    n_rows, top_k = data["retrieved_mask"].shape
    out = np.zeros((n_rows, top_k), dtype=np.float32)
    all_pred = _predict(torch, all_model, data, device, batch_size, mode="all")
    single_model.eval()
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            idx = np.arange(start, end)
            query_text = torch.from_numpy(data["query_text"][idx].astype(np.float32)).to(device)
            query_image = torch.from_numpy(data["query_image"][idx].astype(np.float32)).to(device)
            predict = torch.from_numpy(all_pred[idx].astype(np.float32)).to(device)
            for item_idx in range(top_k):
                candidate_text = torch.from_numpy(data["retrieved_text"][idx, item_idx].astype(np.float32)).to(device)
                candidate_image = torch.from_numpy(data["retrieved_image"][idx, item_idx].astype(np.float32)).to(device)
                candidate_label = torch.from_numpy(data["retrieved_label_scaled"][idx, item_idx].astype(np.float32)).to(device)
                with_pred = single_model(
                    query_text,
                    query_image,
                    candidate_text,
                    candidate_image,
                    candidate_label,
                ).reshape(-1)
                without_pred = single_model(
                    query_text,
                    query_image,
                    query_text,
                    query_image,
                    candidate_label,
                ).reshape(-1)
                score = torch.abs(predict - without_pred) - torch.abs(predict - with_pred)
                valid = torch.from_numpy(data["retrieved_mask"][idx, item_idx].astype(np.float32)).to(device)
                out[idx, item_idx] = (score * valid).detach().cpu().numpy()
    print(f"  [rrcp] mean={out.mean():.5f} max={out.max():.5f} min={out.min():.5f}")
    return out


class _GraphConvolution:
    @staticmethod
    def make(torch, nn, d_model: int, node_count: int):
        class GraphConvolution(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(d_model, d_model))
                self.bias = nn.Parameter(torch.zeros(node_count, d_model))
                nn.init.xavier_uniform_(self.weight)

            def forward(self, feat, adj, mask):
                node_size = adj.size(1)
                adj = torch.clamp(adj, min=0.0)
                eye = torch.eye(node_size, device=adj.device).unsqueeze(0).expand_as(adj)
                adj = adj + eye
                adj = adj * mask.unsqueeze(1)
                degree = torch.pow(adj.sum(-1) + 1e-8, -0.5)
                norm = degree.unsqueeze(-1) * adj * degree.unsqueeze(1)
                pre_sup = torch.matmul(feat, self.weight)
                out = torch.matmul(norm, pre_sup)
                out = out + self.bias[:node_size].unsqueeze(0)
                return torch.tanh(out) * mask.unsqueeze(-1)

        return GraphConvolution()

class _SKAPPAllItemsModel:
    def __new__(cls, text_dim: int, image_dim: int, top_k: int, d_model: int, dropout: float = 0.0):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.text_proj = nn.Linear(text_dim, d_model)
                self.image_proj = nn.Linear(image_dim, d_model)
                self.ret_text_proj = nn.Linear(text_dim, d_model)
                self.ret_image_proj = nn.Linear(image_dim, d_model)
                self.label_embedding = nn.Linear(top_k, d_model)
                self.graph_tt = _GraphConvolution.make(torch, nn, d_model, top_k + 1)
                self.graph_it = _GraphConvolution.make(torch, nn, d_model, top_k + 1)
                self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
                self.predict_1 = nn.Linear(d_model * top_k * 2, d_model)
                self.predict_2 = nn.Linear(d_model * 2, 1)
                self.relu = nn.ReLU()

            def forward(self, query_text, query_image, retrieved_text, retrieved_image, retrieved_label, mask):
                q_text = self.text_proj(query_text).unsqueeze(1)
                q_image = self.image_proj(query_image).unsqueeze(1)
                r_text = self.ret_text_proj(retrieved_text)
                r_image = self.ret_image_proj(retrieved_image)
                text_mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
                img_mask = text_mask
                text_feat = torch.cat([q_text, r_text], dim=1)
                image_feat = torch.cat([q_image, r_image], dim=1)
                edge_tt = _cosine_edge(text_feat, text_mask)
                edge_it = _cosine_edge(image_feat, img_mask)
                graph_tt = self.graph_tt(text_feat, edge_tt, text_mask)[:, 1:, :]
                graph_it = self.graph_it(image_feat, edge_it, img_mask)[:, 1:, :]
                text_out = 0.5 * r_text + 0.5 * (0.7 * graph_tt + 0.3 * graph_it)
                image_out = 0.5 * r_image + 0.5 * (0.7 * graph_tt + 0.3 * graph_it)
                packed = torch.cat([image_out, text_out], dim=1)
                values, _ = self.attn(packed, packed, packed)
                out = values.reshape(values.shape[0], -1)
                out = self.dropout(self.relu(self.predict_1(out)))
                label = self.label_embedding(retrieved_label)
                return self.predict_2(self.dropout(torch.cat([out, label], dim=1)))

        return Model()


class _SKAPPSingleItemModel:
    def __new__(cls, text_dim: int, image_dim: int, top_k: int, d_model: int, dropout: float = 0.0):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.text_proj = nn.Linear(text_dim, d_model)
                self.image_proj = nn.Linear(image_dim, d_model)
                self.ret_text_proj = nn.Linear(text_dim, d_model)
                self.ret_image_proj = nn.Linear(image_dim, d_model)
                self.graph_tt = _GraphConvolution.make(torch, nn, d_model, 2)
                self.graph_it = _GraphConvolution.make(torch, nn, d_model, 2)
                self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
                self.label_embedding = nn.Linear(1, d_model)
                self.predict_1 = nn.Linear(d_model * 2, d_model)
                self.predict_2 = nn.Linear(d_model * 2, 1)
                self.relu = nn.ReLU()

            def forward(self, query_text, query_image, retrieved_text, retrieved_image, retrieved_label):
                q_text = self.text_proj(query_text).unsqueeze(1)
                q_image = self.image_proj(query_image).unsqueeze(1)
                r_text = self.ret_text_proj(retrieved_text).unsqueeze(1)
                r_image = self.ret_image_proj(retrieved_image).unsqueeze(1)
                mask = torch.ones(query_text.shape[0], 2, device=query_text.device)
                text_feat = torch.cat([q_text, r_text], dim=1)
                image_feat = torch.cat([q_image, r_image], dim=1)
                graph_tt = self.graph_tt(text_feat, _cosine_edge(text_feat, mask), mask)[:, 1, :]
                graph_it = self.graph_it(image_feat, _cosine_edge(image_feat, mask), mask)[:, 1, :]
                token = (0.7 * graph_tt + 0.3 * graph_it).unsqueeze(1)
                values, _ = self.attn(token, token, token)
                out = self.dropout(
                    self.relu(self.predict_1(torch.cat([values.squeeze(1), token.squeeze(1)], dim=1)))
                )
                label = self.label_embedding(retrieved_label.unsqueeze(1))
                return self.predict_2(self.dropout(torch.cat([out, label], dim=1)))

        return Model()


class _SKAPPFinalRRCPModel:
    def __new__(
        cls,
        text_dim: int,
        image_dim: int,
        top_k: int,
        d_model: int,
        threshold_of_rrcp: float,
        dropout: float = 0.0,
    ):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.text_proj = nn.Linear(text_dim, d_model)
                self.image_proj = nn.Linear(image_dim, d_model)
                self.ret_text_proj = nn.Linear(text_dim, d_model)
                self.ret_image_proj = nn.Linear(image_dim, d_model)
                self.graph_tt = _GraphConvolution.make(torch, nn, d_model, top_k + 1)
                self.graph_it = _GraphConvolution.make(torch, nn, d_model, top_k + 1)
                self.label_embedding = nn.Linear(top_k, d_model)
                self.predict_1 = nn.Linear(d_model, d_model)
                self.predict_2 = nn.Linear(d_model * 2, 1)
                self.relu = nn.ReLU()

            def forward(self, query_text, query_image, retrieved_text, retrieved_image, retrieved_label, mask, rrcp):
                q_text = self.text_proj(query_text).unsqueeze(1)
                q_image = self.image_proj(query_image).unsqueeze(1)
                r_text = self.ret_text_proj(retrieved_text)
                r_image = self.ret_image_proj(retrieved_image)
                selected = ((rrcp > threshold_of_rrcp).float() * mask).float()
                empty = selected.sum(dim=1, keepdim=True) <= 0
                selected = torch.where(empty, _first_valid(mask), selected)
                weights = torch.clamp(rrcp, min=0.0) * selected
                weights = torch.where(weights.sum(dim=1, keepdim=True) <= 0, selected, weights)
                weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
                graph_mask = torch.cat([torch.ones_like(selected[:, :1]), selected], dim=1)
                text_feat = torch.cat([q_text, r_text], dim=1)
                image_feat = torch.cat([q_image, r_image], dim=1)
                graph_tt = self.graph_tt(
                    text_feat, _cosine_edge(text_feat, graph_mask), graph_mask
                )[:, 1:, :]
                graph_it = self.graph_it(
                    image_feat, _cosine_edge(image_feat, graph_mask), graph_mask
                )[:, 1:, :]
                packed = torch.cat([graph_it, graph_tt], dim=1)
                cxmi = torch.cat([weights, weights], dim=1).unsqueeze(-1)
                context = torch.matmul(packed.transpose(1, 2), cxmi).squeeze(-1)
                out = self.dropout(self.relu(self.predict_1(context)))
                label = self.label_embedding(retrieved_label * selected)
                return self.predict_2(self.dropout(torch.cat([out, label], dim=1)))

        return Model()


def _cosine_edge(feat, mask):
    import torch
    import torch.nn.functional as F

    x = F.normalize(feat, p=2, dim=2) * mask.unsqueeze(-1)
    return torch.bmm(x, x.transpose(1, 2))


def _first_valid(mask):
    import torch

    out = torch.zeros_like(mask)
    has_any = mask.sum(dim=1) > 0
    first_idx = torch.argmax(mask, dim=1)
    out[torch.arange(mask.shape[0], device=mask.device), first_idx] = has_any.float()
    return out


def _inverse_target(y_model: np.ndarray, target: str) -> np.ndarray:
    if TARGET_SPECS[target]["target_transform"] == "log1p":
        return np.expm1(y_model)
    return y_model


def _collect_stage_diagnostics(
    torch,
    model,
    data_by_split: Dict[str, dict],
    splits: Iterable[str],
    device,
    batch_size: int,
    mode: str,
    target: str,
    y_mean: float,
    y_std: float,
) -> dict:
    out = {}
    for split in splits:
        split_data = data_by_split[split]
        pred_scaled = _predict(torch, model, split_data, device, batch_size, mode=mode)
        pred_model = pred_scaled * y_std + y_mean
        item = {
            "scaled": _regression_summary(split_data["y_scaled"], pred_scaled),
            "model_space_prediction": _array_summary(pred_model),
        }
        if "y_model" in split_data and "y_raw" in split_data:
            item.update(
                _prediction_summary(
                    y_raw=split_data["y_raw"],
                    y_model=split_data["y_model"],
                    pred_model=pred_model,
                    target=target,
                    train_y_model=data_by_split["train"]["y_model"]
                    if "train" in data_by_split and "y_model" in data_by_split["train"]
                    else None,
                )
            )
        out[split] = item
    return out


def _prediction_summary(
    y_raw: np.ndarray,
    y_model: np.ndarray,
    pred_model: np.ndarray,
    target: str,
    train_y_model: np.ndarray | None = None,
) -> dict:
    pred_raw = _inverse_target(pred_model, target)
    summary = {
        "model_space": _regression_summary(y_model, pred_model),
        "raw_space": compute_metrics(y_raw, pred_raw, target),
        "target_distribution": _array_summary(y_raw),
        "prediction_distribution": _array_summary(pred_raw),
        "model_prediction_distribution": _array_summary(pred_model),
    }
    if train_y_model is not None and TARGET_SPECS[target]["target_transform"] == "log1p":
        clipped_range = np.clip(pred_model, float(np.min(train_y_model)), float(np.max(train_y_model)))
        clipped_p99 = np.clip(
            pred_model,
            float(np.min(train_y_model)),
            float(np.quantile(train_y_model, 0.99)),
        )
        summary["clipped_to_train_model_range_raw_space"] = compute_metrics(
            y_raw, _inverse_target(clipped_range, target), target
        )
        summary["clipped_to_train_model_p99_raw_space"] = compute_metrics(
            y_raw, _inverse_target(clipped_p99, target), target
        )
    return summary


def _rrcp_summary(rrcp: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    valid = mask > 0
    valid_values = rrcp[valid]
    selected = ((rrcp > threshold) & valid).astype(np.float32)
    selected_count = selected.sum(axis=1)
    empty_rows = int((selected_count == 0).sum())
    return {
        "values": _array_summary(valid_values),
        "positive_ratio": float((valid_values > threshold).mean()) if valid_values.size else 0.0,
        "selected_count": _array_summary(selected_count),
        "empty_rows_after_threshold": empty_rows,
        "mask_mean": float(mask.mean()) if mask.size else 0.0,
    }


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "bias": float(np.mean(error)),
        "target_mean": float(np.mean(y_true)),
        "prediction_mean": float(np.mean(y_pred)),
    }


def _array_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _feature_names(dims: dict) -> List[str]:
    names = []
    names.extend(f"query_text:{idx}" for idx in range(dims["text_dim"]))
    names.extend(f"query_image:{idx}" for idx in range(dims["image_dim"]))
    for pos in range(dims["top_k"]):
        names.extend(f"retrieved_text:{pos}:{idx}" for idx in range(dims["text_dim"]))
        names.extend(f"retrieved_image:{pos}:{idx}" for idx in range(dims["image_dim"]))
        names.append(f"retrieved_label:{pos}")
        names.append(f"rrcp_silver:{pos}")
    return names


def _write_summary(table: pd.DataFrame, path: Path) -> None:
    completed = table[table["status"] == "ok"].copy()
    lines = ["# Baseline Summary", "", "## Completed", ""]
    if completed.empty:
        lines.append("No completed baselines.")
    else:
        cols = [
            "baseline_id",
            "target",
            "model",
            "reference",
            "reproduction_level",
            "val_MAE",
            "val_R2",
            "val_Spearman_rho",
            "test_MAE",
            "test_R2",
            "test_Spearman_rho",
        ]
        lines.append(completed[cols].to_markdown(index=False))
    lines.extend(["", "## Claim Boundaries", ""])
    claim_cols = ["baseline_id", "reference", "reproduction_level", "claim_allowed", "claim_not_allowed"]
    lines.append(table[claim_cols].drop_duplicates(subset=["baseline_id"]).to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_exists(dataset_dir: Path) -> bool:
    return all((dataset_dir / f"{split}.npz").exists() for split in ("train", "val", "test"))


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _set_seed(torch, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_run_dir(config: dict, run_id: str | None) -> Path:
    root = Path(config["output"]["results_dir"])
    if run_id is None:
        run_id = str(config["output"].get("run_id", "01"))
    candidate = root / str(run_id)
    if not candidate.exists():
        return candidate
    try:
        value = int(run_id)
    except ValueError:
        suffix = 2
        while (root / f"{run_id}_{suffix}").exists():
            suffix += 1
        return root / f"{run_id}_{suffix}"
    width = len(str(run_id))
    while True:
        value += 1
        candidate = root / f"{value:0{width}d}"
        if not candidate.exists():
            print(f"[run_id] '{run_id}' already exists -> using '{candidate.name}'")
            return candidate


def _safe_float(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
