"""
Evaluate existing predictions against the MAL July 2025 external-label contract.

The popularity target is cross-platform and scale-shifted:
- model prediction is on the AniList popularity scale
- external label is MAL members

For popularity, prefer rank/log metrics over raw MAE.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "data" / "external_transformed" / "mal_july2025_external_eval_contract.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "external_transformed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions against external MAL labels.")
    parser.add_argument("--predictions-root", required=True, help="Directory containing popularity/ and meanScore/.")
    parser.add_argument("--contract-csv", default=str(DEFAULT_CONTRACT), help="External eval contract CSV.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Prediction split to evaluate.")
    parser.add_argument("--output-prefix", default="external_eval", help="Output file prefix.")
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


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | None]:
    frame = pd.DataFrame(
        {"y_true": pd.to_numeric(y_true, errors="coerce"), "y_pred": pd.to_numeric(y_pred, errors="coerce")}
    ).dropna()
    if frame.empty:
        return {"n": 0, "mae": None, "rmse": None, "spearman": None, "pearson": None}
    err = frame["y_true"].to_numpy(dtype=float) - frame["y_pred"].to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "spearman": None if len(frame) < 2 else round(float(_spearman(frame["y_true"], frame["y_pred"])), 4),
        "pearson": None if len(frame) < 2 else round(float(_pearson(frame["y_true"], frame["y_pred"])), 4),
    }


def _log_mae(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {"y_true": pd.to_numeric(y_true, errors="coerce"), "y_pred": pd.to_numeric(y_pred, errors="coerce")}
    ).dropna()
    if frame.empty:
        return None
    true_log = np.log1p(np.clip(frame["y_true"].to_numpy(dtype=float), 0, None))
    pred_log = np.log1p(np.clip(frame["y_pred"].to_numpy(dtype=float), 0, None))
    return round(float(np.mean(np.abs(true_log - pred_log))), 4)


def _load_predictions(predictions_root: Path, target: str, split: str) -> pd.DataFrame:
    path = predictions_root / target / f"{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path)
    required = {"id", "target", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df.rename(
        columns={
            "target": f"{target}_anilist_target",
            "prediction": f"{target}_prediction",
        }
    )


def main() -> None:
    args = _parse_args()
    predictions_root = Path(args.predictions_root)
    if not predictions_root.is_absolute():
        predictions_root = ROOT / predictions_root
    contract = pd.read_csv(args.contract_csv)

    ready = contract[
        (contract["external_eval_ready"] == True)
        & (contract["split_pre_release_effective"].astype(str) == args.split)
    ].copy()
    ready["id"] = pd.to_numeric(ready["anilist_id"], errors="coerce").astype("Int64")

    pop_pred = _load_predictions(predictions_root, "popularity", args.split)
    score_pred = _load_predictions(predictions_root, "meanScore", args.split)

    pop_eval = ready.merge(pop_pred, on="id", how="inner")
    score_eval = ready.merge(score_pred, on="id", how="inner")

    popularity_metrics = {
        "n": int(len(pop_eval)),
        "spearman_prediction_vs_mal_members": round(
            float(_spearman(pop_eval["popularity_prediction"], pop_eval["external_popularity_members"])), 4
        )
        if len(pop_eval) >= 2
        else None,
        "spearman_prediction_vs_negative_mal_rank": round(
            float(_spearman(pop_eval["popularity_prediction"], -pop_eval["external_popularity_rank"])), 4
        )
        if len(pop_eval) >= 2
        else None,
        "pearson_log_prediction_vs_log_mal_members": round(
            float(
                _pearson(
                    np.log1p(np.clip(pop_eval["popularity_prediction"], 0, None)),
                    np.log1p(np.clip(pop_eval["external_popularity_members"], 0, None)),
                )
            ),
            4,
        )
        if len(pop_eval) >= 2
        else None,
        "log_mae_prediction_vs_mal_members": _log_mae(
            pop_eval["external_popularity_members"], pop_eval["popularity_prediction"]
        ),
        "scale_note": "Raw MAE is intentionally omitted: AniList popularity and MAL members use different count scales.",
    }

    mean_score_metrics = _regression_metrics(
        score_eval["external_score_0_100"], score_eval["meanScore_prediction"]
    )
    label_alignment = {
        "anilist_popularity_vs_mal_members_spearman": round(
            float(_spearman(ready["anilist_popularity"], ready["external_popularity_members"])), 4
        ),
        "anilist_meanScore_vs_mal_score100": round(
            float(_spearman(ready["anilist_meanScore"], ready["external_score_0_100"])), 4
        ),
    }

    merged_cols = [
        "id",
        "mal_id",
        "title_name",
        "title_romaji",
        "external_popularity_members",
        "external_popularity_rank",
        "external_score_0_100",
        "anilist_popularity",
        "anilist_meanScore",
    ]
    detail = ready[merged_cols].merge(
        pop_pred[["id", "popularity_prediction"]],
        on="id",
        how="left",
    ).merge(
        score_pred[["id", "meanScore_prediction"]],
        on="id",
        how="left",
    )

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = DEFAULT_OUT_DIR / f"{args.output_prefix}_{args.split}_details.csv"
    metrics_path = DEFAULT_OUT_DIR / f"{args.output_prefix}_{args.split}_metrics.json"
    detail.to_csv(detail_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "predictions_root": predictions_root.as_posix(),
        "contract_csv": Path(args.contract_csv).as_posix(),
        "ready_contract_rows_for_split": int(len(ready)),
        "popularity": popularity_metrics,
        "meanScore": mean_score_metrics,
        "external_label_alignment": label_alignment,
        "outputs": {
            "details_csv": detail_path.as_posix(),
            "metrics_json": metrics_path.as_posix(),
        },
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
