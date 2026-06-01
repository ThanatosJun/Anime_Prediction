from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.experiment_common.metrics import compute_metrics
from src.reference_baseline_branch.c3_source_faithful_data import build_source_faithful_npz
from src.reference_baseline_branch.skapp_source_faithful_models import (
    SourceFaithfulAllItemsModel,
    SourceFaithfulFinalModel,
    SourceFaithfulSingleItemModel,
)


TARGET_SPECS = {
    "popularity": {"retrieved_label_idx": 0, "target_transform": "log1p"},
    "meanScore": {"retrieved_label_idx": 1, "target_transform": "identity"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run source-aligned staged C3 pipeline: all-items -> single-item -> RRCP -> final."
    )
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
    )
    parser.add_argument("--target", choices=tuple(TARGET_SPECS), default="popularity")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--force-rebuild-dataset", action="store_true")
    parser.add_argument("--run-id", default="v2_source_exact_c3")
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Retrieval size for dataset tensors (source paper default: 500).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load saved checkpoints and recompute metrics/predictions.",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    device = _resolve_device(args.device)
    _set_seed(2024)

    dataset_dir = Path(config["data"].get("skapp_full_dataset_dir", ".exp/baseline/skapp_full/dataset_v2"))
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only and (args.force_rebuild_dataset or not _dataset_exists(dataset_dir)):
        build_source_faithful_npz(
            config=config,
            dataset_dir=dataset_dir,
            top_k=args.top_k,
            device=device.type,
        )

    run_root = Path(config["output"]["results_dir"]) / str(args.run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    data = {
        split: _load_split(dataset_dir / f"{split}.npz", target=args.target)
        for split in ("train", "val", "test")
    }
    y_mean, y_std = _fit_scaler(data["train"]["y_model"])
    for split_data in data.values():
        split_data["y_scaled"] = ((split_data["y_model"] - y_mean) / y_std).astype(np.float32)
        split_data["retrieved_label_scaled"] = ((split_data["retrieved_label"] - y_mean) / y_std).astype(np.float32)

    text_dim = int(data["train"]["query_text"].shape[1])
    image_dim = int(data["train"]["query_image"].shape[1])
    top_k = int(data["train"]["retrieved_text"].shape[1])
    d_model = 768

    if args.eval_only:
        _run_eval_only(
            run_root=run_root,
            data=data,
            target=args.target,
            run_id=args.run_id,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
        )
        return

    # Stage 1: all-items (source: RRCP/train_all_item.py)
    all_model_path = run_root / "model_all_items.pth"
    if all_model_path.exists():
        print("[resume] load stage1 all-items model")
        all_model = torch.load(all_model_path, map_location=device, weights_only=False)
        all_model.eval()
    else:
        all_model = SourceFaithfulAllItemsModel(
            text_dim=text_dim,
            image_dim=image_dim,
            top_k=top_k,
            d_model=d_model,
            strict_source=True,
        ).to(device)
        _train_stage_model(
            model=all_model,
            train=data["train"],
            val=data["val"],
            device=device,
            mode="all",
            batch_size=64,
            max_epochs=1000,
            patience=10,
            lr=1e-4,
            optimizer_kind="adam",
        )
        torch.save(all_model, all_model_path)

    # Stage 2: single-item/dissembled (source: RRCP/train_single_item.py)
    single_train = _make_disassembled_data(data["train"])
    single_val = _make_disassembled_data(data["val"])
    single_model_path = run_root / "model_single_item.pth"
    if single_model_path.exists():
        print("[resume] load stage2 single-item model")
        single_model = torch.load(single_model_path, map_location=device, weights_only=False)
        single_model.eval()
    else:
        single_model = SourceFaithfulSingleItemModel(
            text_dim=text_dim,
            image_dim=image_dim,
            d_model=d_model,
            strict_source=True,
        ).to(device)
        _train_stage_model(
            model=single_model,
            train=single_train,
            val=single_val,
            device=device,
            mode="single",
            batch_size=1024,
            max_epochs=1000,
            patience=10,
            lr=1e-4,
            optimizer_kind="adam",
        )
        torch.save(single_model, single_model_path)

    # Stage 3: RRCP_silver (source: RRCP/RRCP.py)
    for split in ("train", "val", "test"):
        rrcp_path = run_root / f"rrcp_silver_{split}.npz"
        if rrcp_path.exists():
            print(f"[resume] load stage3 rrcp_silver {split}")
            data[split]["rrcp_silver"] = np.load(rrcp_path)["rrcp_silver"].astype(np.float32)
        else:
            data[split]["rrcp_silver"] = _compute_rrcp_silver(
                all_model=all_model,
                single_model=single_model,
                data=data[split],
                device=device,
                batch_size=128,
            )
            np.savez_compressed(
                rrcp_path,
                ids=data[split]["ids"],
                rrcp_silver=data[split]["rrcp_silver"],
            )

    # Stage 4: final RRCP prediction (source: RRCP_prediction_variable_lenth.py + graph_attention.py)
    final_model_path = run_root / "model_final.pth"
    if final_model_path.exists():
        print("[resume] load stage4 final model")
        final_model = torch.load(final_model_path, map_location=device, weights_only=False)
        final_model.eval()
    else:
        final_model = SourceFaithfulFinalModel(
            text_dim=text_dim,
            image_dim=image_dim,
            top_k=top_k,
            d_model=d_model,
            threshold_of_rrcp=0.0,
            strict_source=True,
        ).to(device)
        _train_stage_model(
            model=final_model,
            train=data["train"],
            val=data["val"],
            device=device,
            mode="final",
            batch_size=64,
            max_epochs=1000,
            patience=5,
            lr=1e-5,
            optimizer_kind="adam",
        )
        torch.save(final_model, final_model_path)

    _write_results(
        run_root=run_root,
        final_model=final_model,
        data=data,
        target=args.target,
        run_id=args.run_id,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
    )


def _run_eval_only(
    run_root: Path,
    data: dict,
    target: str,
    run_id: str,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> None:
    final_model_path = run_root / "model_final.pth"
    if not final_model_path.exists():
        raise FileNotFoundError(f"Missing final model checkpoint: {final_model_path}")
    print("[eval-only] load stage4 final model")
    final_model = torch.load(final_model_path, map_location=device, weights_only=False)
    final_model.eval()
    for split in ("train", "val", "test"):
        rrcp_path = run_root / f"rrcp_silver_{split}.npz"
        if not rrcp_path.exists():
            raise FileNotFoundError(f"Missing RRCP cache: {rrcp_path}")
        data[split]["rrcp_silver"] = np.load(rrcp_path)["rrcp_silver"].astype(np.float32)
    _write_results(
        run_root=run_root,
        final_model=final_model,
        data=data,
        target=target,
        run_id=run_id,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
    )


def _write_results(
    run_root: Path,
    final_model: torch.nn.Module,
    data: dict,
    target: str,
    run_id: str,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> None:
    train_y_model = data["train"]["y_model"]
    val_pred = _predict(final_model, data["val"], device=device, batch_size=64, mode="final")
    test_pred = _predict(final_model, data["test"], device=device, batch_size=64, mode="final")
    val_pred_model = _denorm_and_clip(val_pred, y_mean, y_std, train_y_model)
    test_pred_model = _denorm_and_clip(test_pred, y_mean, y_std, train_y_model)
    val_pred_raw = _inverse_target(val_pred_model, target)
    test_pred_raw = _inverse_target(test_pred_model, target)

    val_metrics = compute_metrics(data["val"]["y_raw"], val_pred_raw, target)
    test_metrics = compute_metrics(data["test"]["y_raw"], test_pred_raw, target)
    out = {
        "target": target,
        "run_id": run_id,
        "pipeline": "source_exact_staged",
        "val": val_metrics,
        "test": test_metrics,
        "notes": (
            "Only project input/output domain mapping differs from source SKAPP data/task. "
            "Predictions are clipped to train-set model-space target range before inverse transform."
        ),
    }
    with open(run_root / f"metrics_{target}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    pd.DataFrame(
        {"id": data["test"]["ids"], "target": data["test"]["y_raw"], "prediction": test_pred_raw}
    ).to_csv(run_root / f"test_predictions_{target}.csv", index=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def _denorm_and_clip(
    pred_scaled: np.ndarray,
    y_mean: float,
    y_std: float,
    train_y_model: np.ndarray,
) -> np.ndarray:
    pred_model = pred_scaled.astype(np.float64) * y_std + y_mean
    lo = float(np.min(train_y_model))
    hi = float(np.max(train_y_model))
    return np.clip(pred_model, lo, hi).astype(np.float32)


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_exists(dataset_dir: Path) -> bool:
    return all((dataset_dir / f"{split}.npz").exists() for split in ("train", "val", "test"))


def _fit_scaler(y: np.ndarray) -> tuple[float, float]:
    mean = float(y.mean())
    std = float(y.std())
    if std <= 1e-8:
        std = 1.0
    return mean, std


def _load_split(path: Path, target: str) -> dict:
    item = dict(np.load(path))
    spec = TARGET_SPECS[target]
    raw = item[target].astype(np.float32)
    if spec["target_transform"] == "log1p":
        y_model = np.log1p(np.clip(raw, 0, None)).astype(np.float32)
    else:
        y_model = raw.astype(np.float32)
    retrieved_label = item["retrieved_labels"][:, :, spec["retrieved_label_idx"]].astype(np.float32)
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


def _batch(data: dict, idx: np.ndarray, device: torch.device, mode: str):
    y = torch.from_numpy(data["y_scaled"][idx].astype(np.float32)).to(device)
    q_text = torch.from_numpy(data["query_text"][idx].astype(np.float32)).to(device)
    q_image = torch.from_numpy(data["query_image"][idx].astype(np.float32)).to(device)
    if mode == "single":
        r_text = torch.from_numpy(data["retrieved_text_one"][idx].astype(np.float32)).to(device)
        r_image = torch.from_numpy(data["retrieved_image_one"][idx].astype(np.float32)).to(device)
        r_label = torch.from_numpy(data["retrieved_label_one"][idx].astype(np.float32)).to(device)
        return q_text, q_image, r_text, r_image, r_label, y
    r_text = torch.from_numpy(data["retrieved_text"][idx].astype(np.float32)).to(device)
    r_image = torch.from_numpy(data["retrieved_image"][idx].astype(np.float32)).to(device)
    r_label = torch.from_numpy(data["retrieved_label_scaled"][idx].astype(np.float32)).to(device)
    r_mask = torch.from_numpy(data["retrieved_mask"][idx].astype(np.float32)).to(device)
    if mode == "final":
        rrcp = torch.from_numpy(data["rrcp_silver"][idx].astype(np.float32)).to(device)
        return q_text, q_image, r_text, r_image, r_label, r_mask, rrcp, y
    return q_text, q_image, r_text, r_image, r_label, r_mask, y


def _train_stage_model(
    model: torch.nn.Module,
    train: dict,
    val: dict,
    device: torch.device,
    mode: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    optimizer_kind: str,
) -> None:
    model.train()
    loss_fn = torch.nn.MSELoss()
    if optimizer_kind == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best = np.inf
    stale = 0
    best_state = None
    rng = np.random.default_rng(42)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train["y_scaled"]))
        train_losses = []
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            batch = _batch(train, idx, device=device, mode=mode)
            optimizer.zero_grad(set_to_none=True)
            pred = model(*batch[:-1]).reshape(-1)
            loss = loss_fn(pred, batch[-1])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for s in range(0, len(val["y_scaled"]), batch_size):
                idx = np.arange(s, min(s + batch_size, len(val["y_scaled"])))
                batch = _batch(val, idx, device=device, mode=mode)
                pred = model(*batch[:-1]).reshape(-1)
                val_losses.append(float(loss_fn(pred, batch[-1]).detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else np.inf
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        print(f"[{mode}] epoch={epoch} train={train_loss:.5f} val={val_loss:.5f}")
        if val_loss < best - 1e-5:
            best = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale > patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


def _predict(model, data: dict, device: torch.device, batch_size: int, mode: str) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(data["ids"]), batch_size):
            idx = np.arange(s, min(s + batch_size, len(data["ids"])))
            batch = _batch(data, idx, device=device, mode=mode)
            pred = model(*batch[:-1]).reshape(-1).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0)


def _compute_rrcp_silver(
    all_model,
    single_model,
    data: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n_rows, top_k = data["retrieved_mask"].shape
    out = np.zeros((n_rows, top_k), dtype=np.float32)
    all_pred = _predict(all_model, data, device=device, batch_size=batch_size, mode="all")
    with torch.no_grad():
        for s in range(0, n_rows, batch_size):
            idx = np.arange(s, min(s + batch_size, n_rows))
            q_text = torch.from_numpy(data["query_text"][idx].astype(np.float32)).to(device)
            q_image = torch.from_numpy(data["query_image"][idx].astype(np.float32)).to(device)
            predict = torch.from_numpy(all_pred[idx].astype(np.float32)).to(device)
            for j in range(top_k):
                c_text = torch.from_numpy(data["retrieved_text"][idx, j].astype(np.float32)).to(device)
                c_img = torch.from_numpy(data["retrieved_image"][idx, j].astype(np.float32)).to(device)
                c_lab = torch.from_numpy(data["retrieved_label_scaled"][idx, j].astype(np.float32)).to(device)
                with_ret = single_model(q_text, q_image, c_text, c_img, c_lab).reshape(-1)
                without_ret = single_model(q_text, q_image, q_text, q_image, c_lab).reshape(-1)
                score = torch.abs(predict - without_ret) - torch.abs(predict - with_ret)
                valid = torch.from_numpy(data["retrieved_mask"][idx, j].astype(np.float32)).to(device)
                out[idx, j] = (score * valid).detach().cpu().numpy()
    return out


def _inverse_target(y_model: np.ndarray, target: str) -> np.ndarray:
    if TARGET_SPECS[target]["target_transform"] == "log1p":
        y_model = np.clip(y_model, 0.0, 20.0)
        with np.errstate(over="ignore"):
            return np.expm1(y_model)
    return y_model


if __name__ == "__main__":
    main()
