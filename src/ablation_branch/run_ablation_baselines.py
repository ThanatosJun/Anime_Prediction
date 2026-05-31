from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/ablation_branch/configs/ablation_baselines.yaml",
        help="Path to ablation baseline config YAML.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[ablation] runner scaffold is ready.")
    print(f"[ablation] config: {config_path}")
    print("[ablation] planned groups:")
    for group in config.get("ablation_groups", []):
        print(f"  - {group['id']}: {group['description']}")
    print("[ablation] implementation will reuse src/experiment_common features and metrics.")


if __name__ == "__main__":
    main()
