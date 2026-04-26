import yaml


def load_config(path: str = "src/fussion_branch/configs/fusion_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
