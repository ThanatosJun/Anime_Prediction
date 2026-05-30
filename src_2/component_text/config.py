import yaml
from pathlib import Path


def load_config(config_path: str = 'text_encoder_config.yaml') -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)
