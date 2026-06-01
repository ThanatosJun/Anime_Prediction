"""
Entry point: E2E fine-tune FusionMLPE2E (with trainable text encoder backbone).

Usage:
    python -m src.fussion_branch.run_train_e2e
    python -m src.fussion_branch.run_train_e2e --target popularity
    python -m src.fussion_branch.run_train_e2e --config src/fussion_branch/configs/fusion_config_e2e.yaml
"""
import argparse
import json
from pathlib import Path

import yaml

from src.fussion_branch.utilities.config import load_config, resolve_run_id
from fussion_branch.test_pipeline.train_e2e import train_one_target_e2e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["popularity", "meanScore", "both"],
        default=None,
    )
    parser.add_argument(
        "--config",
        default="src/fussion_branch/configs/fusion_config_e2e.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_run_id(config)

    run_dir = Path(config["output"]["results_dir"]) / config["output"]["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    if args.target is not None:
        targets = ["popularity", "meanScore"] if args.target == "both" else [args.target]
    else:
        targets = config.get("active_targets", ["popularity", "meanScore"])

    all_results = {}
    for target in targets:
        print(f"\n{'='*60}")
        print(f"  Training E2E: {target}")
        print(f"{'='*60}")
        results = train_one_target_e2e(config, target)
        all_results[target] = results

    print("\n" + "="*60)
    print("  Final Results")
    print("="*60)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
