from pathlib import Path

import yaml


def load_config(path: str = "src/fussion_branch/configs/fusion_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_run_id(config: dict) -> dict:
    """If results_dir/run_id already exists, auto-increment to the next available id."""
    results_dir = Path(config["output"]["results_dir"])
    run_id = config["output"]["run_id"]

    if not (results_dir / run_id).exists():
        return config

    # find next available numeric id
    n = int(run_id)
    while (results_dir / f"{n:02d}").exists():
        n += 1
    new_id = f"{n:02d}"
    config["output"]["run_id"] = new_id
    print(f"[run_id] '{run_id}' already exists → using '{new_id}'")
    return config
