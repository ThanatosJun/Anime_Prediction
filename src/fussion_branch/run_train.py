"""
Entry point: trains Model A (popularity) and Model B (meanScore) independently.

Usage:
    python -m src.fussion_branch.run_train                      # uses active_targets from config
    python -m src.fussion_branch.run_train --target popularity  # CLI overrides config
    python -m src.fussion_branch.run_train --target meanScore
    python -m src.fussion_branch.run_train --target both
"""
import argparse
import json
from pathlib import Path

import yaml

from src.fussion_branch.utilities.config import load_config, resolve_run_id
from src.fussion_branch.utilities.summarize_experiments import collect
from src.fussion_branch.fussion_training.train import train_one_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["popularity", "meanScore", "both"],
        default=None,
        help="Override active_targets in config",
    )
    parser.add_argument(
        "--config",
        default="src/fussion_branch/configs/fusion_config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_run_id(config)

    # save config snapshot to run directory
    run_dir = Path(config["output"]["results_dir"]) / config["output"]["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    # CLI --target overrides config; fall back to active_targets in config
    if args.target is not None:
        targets = ["popularity", "meanScore"] if args.target == "both" else [args.target]
    else:
        targets = config.get("active_targets", ["popularity", "meanScore"])

    all_results = {}
    for target in targets:
        print(f"\n{'='*60}")
        print(f"  Training: {target}")
        print(f"{'='*60}")
        results = train_one_target(config, target)
        all_results[target] = results

    print("\n" + "="*60)
    print("  Final Results")
    print("="*60)
    print(json.dumps(all_results, indent=2))

    out_csv = Path(".exp/fussion/experiments_summary.csv")
    collect().to_csv(out_csv, index=False)
    print(f"\n[summary] updated → {out_csv}")


if __name__ == "__main__":
    main()
