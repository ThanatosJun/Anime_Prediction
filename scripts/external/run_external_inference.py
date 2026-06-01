"""
Run src_2 models on a prepared external split and join MAL labels.

This helper intentionally does not modify src_2/evaluate.py because external
metrics differ from internal AniList metrics, especially popularity scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_2" / "fussion_training"))

from dataset import AnimeDataset, denormalize_target  # noqa: E402
from meta_encoder import MetaEncoder  # noqa: E402
from model import FusionModel, make_model_config  # noqa: E402


DEFAULT_TARGETS = ["popularity", "meanScore"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external MAL-only inference.")
    parser.add_argument("--config", default="src_2/fussion_configs.yaml")
    parser.add_argument("--split", required=True, help="External split name, e.g. mal2025_dual_local_ready.")
    parser.add_argument("--id-map", default=None, help="Sidecar id map CSV. Defaults to data/external_transformed/<split>_id_map.csv.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, choices=DEFAULT_TARGETS)
    parser.add_argument("--run-id", default=None, help="Override output.run_id from the config, e.g. 02.")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    frame = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(frame) < 2:
        return None
    return float(frame["a"].rank(method="average").corr(frame["b"].rank(method="average")))


def _pearson(a: pd.Series, b: pd.Series) -> float | None:
    frame = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(frame) < 2:
        return None
    return float(frame["a"].corr(frame["b"]))


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    frame = pd.DataFrame(
        {"y_true": pd.to_numeric(y_true, errors="coerce"), "y_pred": pd.to_numeric(y_pred, errors="coerce")}
    ).dropna()
    if frame.empty:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "spearman": None,
            "pearson": None,
            "acc_within_10pt": None,
        }
    err = frame["y_true"].to_numpy(dtype=float) - frame["y_pred"].to_numpy(dtype=float)
    denom = float(np.sum((frame["y_true"].to_numpy(dtype=float) - frame["y_true"].mean()) ** 2))
    return {
        "n": int(len(frame)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "r2": round(float(1.0 - np.sum(err**2) / denom), 4) if denom > 0 else None,
        "spearman": round(float(_spearman(frame["y_true"], frame["y_pred"])), 4) if len(frame) >= 2 else None,
        "pearson": round(float(_pearson(frame["y_true"], frame["y_pred"])), 4) if len(frame) >= 2 else None,
        "acc_within_10pt": round(float(np.mean(np.abs(err) < 10)), 4),
    }


def _popularity_metrics(y_true: pd.Series, y_pred: pd.Series, rank: pd.Series | None = None) -> dict:
    frame = pd.DataFrame(
        {"y_true": pd.to_numeric(y_true, errors="coerce"), "y_pred": pd.to_numeric(y_pred, errors="coerce")}
    ).dropna()
    if frame.empty:
        return {
            "n": 0,
            "spearman_prediction_vs_mal_members": None,
            "spearman_prediction_vs_negative_mal_rank": None,
            "pearson_log_prediction_vs_log_mal_members": None,
            "log_mae_prediction_vs_mal_members": None,
            "log_r2_prediction_vs_mal_members": None,
            "factor_acc_2x_prediction_vs_mal_members": None,
            "raw_mae_prediction_vs_mal_members": None,
            "scale_note": "Raw MAE is diagnostic only because AniList popularity predictions and MAL members use different count scales.",
        }
    true_log = np.log1p(np.clip(frame["y_true"].to_numpy(dtype=float), 0, None))
    pred_log = np.log1p(np.clip(frame["y_pred"].to_numpy(dtype=float), 0, None))
    log_err = true_log - pred_log
    denom = float(np.sum((true_log - np.mean(true_log)) ** 2))
    negative_rank_spearman = None
    if rank is not None:
        negative_rank_spearman = round(float(_spearman(y_pred, -rank)), 4)
    return {
        "n": int(len(frame)),
        "spearman_prediction_vs_mal_members": round(float(_spearman(y_pred, y_true)), 4) if len(frame) >= 2 else None,
        "spearman_prediction_vs_negative_mal_rank": negative_rank_spearman,
        "pearson_log_prediction_vs_log_mal_members": round(float(_pearson(pd.Series(pred_log), pd.Series(true_log))), 4)
        if len(frame) >= 2
        else None,
        "log_mae_prediction_vs_mal_members": round(float(np.mean(np.abs(log_err))), 4),
        "log_r2_prediction_vs_mal_members": round(float(1.0 - np.sum(log_err**2) / denom), 4) if denom > 0 else None,
        "factor_acc_2x_prediction_vs_mal_members": round(float(np.mean(np.abs(log_err) < np.log(2))), 4),
        "raw_mae_prediction_vs_mal_members": round(
            float(np.mean(np.abs(frame["y_true"].to_numpy(dtype=float) - frame["y_pred"].to_numpy(dtype=float)))),
            4,
        ),
        "scale_note": "Raw MAE is diagnostic only because AniList popularity predictions and MAL members use different count scales.",
    }


@torch.no_grad()
def _predict_target(target: str, split: str, config: dict, meta_encoder: MetaEncoder) -> pd.DataFrame:
    cfg_train = config["training"]
    cfg_out = config["output"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg_out["run_dir"]) / cfg_out["run_id"] / target
    ckpt = run_dir / "best_model.pt"
    scaler_path = run_dir / "target_scaler.json"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Target scaler not found: {scaler_path}")

    target_scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    ds = AnimeDataset(split, config, meta_encoder, target=target, target_scaler=target_scaler)
    loader = DataLoader(ds, batch_size=cfg_train["batch_size"] * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = FusionModel(make_model_config(config, target)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    preds, ids = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        preds.append(model(batch).cpu().numpy())
        ids.extend(batch["id"].cpu().tolist())
    pred_orig = denormalize_target(np.concatenate(preds), target_scaler)
    return pd.DataFrame({"id": ids, f"prediction_{target}": pred_orig})


def main() -> None:
    args = _parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.run_id:
        config["output"]["run_id"] = args.run_id

    meta_encoder_path = Path(config["data"]["meta_encoder_path"])
    if not meta_encoder_path.is_absolute():
        meta_encoder_path = ROOT / meta_encoder_path
    if not meta_encoder_path.exists():
        raise FileNotFoundError(f"MetaEncoder not found: {meta_encoder_path}. Run or copy the trained meta encoder first.")
    meta_encoder = MetaEncoder.load(str(meta_encoder_path))

    id_map_path = Path(args.id_map) if args.id_map else ROOT / "data" / "external_transformed" / f"{args.split}_id_map.csv"
    if not id_map_path.is_absolute():
        id_map_path = ROOT / id_map_path
    detail = pd.read_csv(id_map_path)
    for target in args.targets:
        detail = detail.merge(_predict_target(target, args.split, config, meta_encoder), on="id", how="left")

    metrics = {}
    if "prediction_popularity" in detail.columns:
        metrics["popularity"] = _popularity_metrics(
            detail["external_popularity_members"],
            detail["prediction_popularity"],
            detail["external_popularity_rank"] if "external_popularity_rank" in detail.columns else None,
        )
    if "prediction_meanScore" in detail.columns:
        metrics["meanScore"] = _regression_metrics(detail["external_score_0_100"], detail["prediction_meanScore"])

    out_dir = ROOT / "data" / "external_transformed"
    out_prefix = args.output_prefix or f"{config['output']['run_id']}_{args.split}_external"
    detail_path = out_dir / f"{out_prefix}_predictions.csv"
    metrics_path = out_dir / f"{out_prefix}_metrics.json"
    detail.to_csv(detail_path, index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "run_id": config["output"]["run_id"],
        "id_map": id_map_path.relative_to(ROOT).as_posix(),
        "metrics": metrics,
        "outputs": {
            "predictions_csv": detail_path.relative_to(ROOT).as_posix(),
            "metrics_json": metrics_path.relative_to(ROOT).as_posix(),
        },
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
