from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.experiment_common.features import BaselineFeatureStore, SplitFeatures
from src.experiment_common.metrics import compute_metrics, inverse_target, transform_target
from src.reference_baseline_branch.sklearn_models import make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/reference_baseline_branch/configs/reference_baselines.yaml",
        help="Path to reference baseline config YAML.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional single baseline id to run.",
    )
    parser.add_argument(
        "--target",
        choices=["popularity", "meanScore"],
        default=None,
        help="Optional single target to run.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Allow running a disabled baseline when --baseline is supplied.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    store = BaselineFeatureStore(config)
    store.load_metadata()

    baselines = _select_baselines(config, args.baseline, args.include_disabled)
    targets = [args.target] if args.target else list(config["targets"].keys())

    rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        for target in targets:
            print(f"\n[baseline] {baseline['id']}  target={target}")
            try:
                row = _run_one(config, store, baseline, target, run_dir)
            except Exception as exc:
                row = _with_empty_metrics(
                    {
                        "baseline_id": baseline["id"],
                        "target": target,
                        "feature_set": baseline.get("feature_set", ""),
                        "model": baseline.get("model", ""),
                        "reference": baseline.get("reference", ""),
                        "reproduction_level": baseline.get("reproduction_level", ""),
                        "paper_supported_component": baseline.get("paper_supported_component", ""),
                        "project_adaptation_component": baseline.get("project_adaptation_component", ""),
                        "claim_allowed": baseline.get("claim_allowed", ""),
                        "claim_not_allowed": baseline.get("claim_not_allowed", ""),
                        "status": "failed",
                        "notes": f"{type(exc).__name__}: {exc}",
                    }
                )
                error_dir = run_dir / "errors"
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / f"{baseline['id']}_{target}.txt"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
            rows.append(row)
            print(f"  status={row['status']}  notes={row.get('notes', '')}")

    table = pd.DataFrame(rows)
    table_path = run_dir / "baseline_results.csv"
    table.to_csv(table_path, index=False)
    with open(run_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    _write_summary(table, run_dir / "baseline_summary.md", config)
    print(f"\n[done] results saved to {run_dir}")
    print(table_path)


def _run_one(
    config: dict,
    store: BaselineFeatureStore,
    baseline: dict,
    target: str,
    run_dir: Path,
) -> Dict[str, Any]:
    feature_set_id = baseline["feature_set"]
    feature_set = config["feature_sets"][feature_set_id]
    split_data, feature_names, missing = store.build(feature_set, target)

    base_row = {
        "baseline_id": baseline["id"],
        "target": target,
        "feature_set": feature_set_id,
        "model": baseline["model"],
        "reference": baseline.get("reference", ""),
        "reproduction_level": baseline.get("reproduction_level", ""),
        "paper_supported_component": baseline.get("paper_supported_component", ""),
        "project_adaptation_component": baseline.get("project_adaptation_component", ""),
        "claim_allowed": baseline.get("claim_allowed", ""),
        "claim_not_allowed": baseline.get("claim_not_allowed", ""),
        "status": "ok",
        "notes": "",
    }

    if missing:
        base_row.update({"status": "skipped", "notes": missing})
        return _with_empty_metrics(base_row)

    try:
        model = make_model(baseline["model"], baseline.get("params", {}))
    except ImportError as exc:
        base_row.update({"status": "skipped", "notes": str(exc)})
        return _with_empty_metrics(base_row)

    train = split_data["train"]
    log_transform = bool(config["targets"][target].get("log_transform", False))
    y_train = transform_target(train.y_raw, log_transform)

    if baseline["model"] != "mean" and train.x is None:
        base_row.update({"status": "skipped", "notes": "Model requires features but feature_set has none"})
        return _with_empty_metrics(base_row)

    if baseline["model"] == "mean":
        model.fit(None, y_train)
    else:
        model.fit(train.x, y_train)

    row = dict(base_row)
    row["n_train"] = int(len(train.y_raw))
    row["n_features"] = int(0 if train.x is None else train.x.shape[1])

    pred_dir = run_dir / "predictions" / baseline["id"] / target
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        data = split_data[split]
        y_pred_model = _predict(model, baseline["model"], data)
        y_pred = inverse_target(y_pred_model, log_transform)
        metrics = compute_metrics(data.y_raw, y_pred, target)
        for key, value in metrics.items():
            row[f"{split}_{key}"] = value
        row[f"n_{split}"] = int(len(data.y_raw))
        pd.DataFrame(
            {
                "id": data.ids,
                "target": data.y_raw,
                "prediction": y_pred,
            }
        ).to_csv(pred_dir / f"{split}_predictions.csv", index=False)

    with open(pred_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)

    return row


def _predict(model, model_name: str, data: SplitFeatures) -> np.ndarray:
    if model_name == "mean":
        return model.predict_n(len(data.y_raw))
    return model.predict(data.x)


def _with_empty_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    for split in ("val", "test"):
        for metric in ("MAE", "RMSE", "R2", "Spearman_rho", "Pearson_r", "log_MAE"):
            row[f"{split}_{metric}"] = ""
        row[f"n_{split}"] = ""
    row.setdefault("n_train", "")
    row.setdefault("n_features", "")
    return row


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_run_dir(config: dict) -> Path:
    out_cfg = config["output"]
    root = Path(out_cfg["results_dir"])
    run_id = str(out_cfg["run_id"])
    candidate = root / run_id
    if not candidate.exists():
        return candidate

    width = len(run_id)
    try:
        value = int(run_id)
    except ValueError:
        suffix = 2
        while (root / f"{run_id}_{suffix}").exists():
            suffix += 1
        return root / f"{run_id}_{suffix}"

    while candidate.exists():
        value += 1
        candidate = root / f"{value:0{width}d}"
    print(f"[run_id] '{run_id}' already exists -> using '{candidate.name}'")
    return candidate


def _select_baselines(
    config: dict,
    baseline_id: Optional[str],
    include_disabled: bool,
) -> List[dict]:
    baselines = config["baselines"]
    if baseline_id:
        matches = [item for item in baselines if item["id"] == baseline_id]
        if not matches:
            raise ValueError(f"Unknown baseline id: {baseline_id}")
        if not include_disabled and not matches[0].get("enabled", True):
            raise ValueError(f"Baseline is disabled in config: {baseline_id}")
        return matches
    return [item for item in baselines if item.get("enabled", True)]


def _write_summary(table: pd.DataFrame, path: Path, config: dict) -> None:
    ok = table[table["status"] == "ok"]
    skipped = table[table["status"] != "ok"]

    lines = [
        "# Baseline Summary",
        "",
        f"Run notes: {config['output'].get('notes', '')}",
        "",
        "## Completed",
        "",
    ]
    if ok.empty:
        lines.append("No completed baselines.")
    else:
        cols = [
            "baseline_id",
            "target",
            "feature_set",
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
        lines.append(ok[cols].to_markdown(index=False))

    lines.extend(["", "## Skipped", ""])
    if skipped.empty:
        lines.append("No skipped baselines.")
    else:
        lines.append(skipped[["baseline_id", "target", "model", "reproduction_level", "notes"]].to_markdown(index=False))

    lines.extend(["", "## Claim Boundaries", ""])
    claim_cols = [
        "baseline_id",
        "reference",
        "reproduction_level",
        "claim_allowed",
        "claim_not_allowed",
    ]
    claim_table = table[claim_cols].drop_duplicates(subset=["baseline_id"])
    lines.append(claim_table.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
