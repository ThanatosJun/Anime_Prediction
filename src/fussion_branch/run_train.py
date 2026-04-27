"""
Entry point: trains Model A (popularity) and Model B (meanScore) independently.

Usage:
    python -m src.fussion_branch.run_train
    python -m src.fussion_branch.run_train --target popularity
    python -m src.fussion_branch.run_train --target meanScore
"""
import argparse
import json

from src.fussion_branch.utilities.config import load_config
from src.fussion_branch.fussion_training.train import train_one_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["popularity", "meanScore", "both"],
        default="both",
        help="Which target to train (default: both)",
    )
    parser.add_argument(
        "--config",
        default="src/fussion_branch/configs/fusion_config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config  = load_config(args.config)
    targets = ["popularity", "meanScore"] if args.target == "both" else [args.target]

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


if __name__ == "__main__":
    main()
