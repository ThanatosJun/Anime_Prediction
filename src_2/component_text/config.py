import yaml
from pathlib import Path

_DEFAULT_CFG = Path(__file__).parent / 'text_encoder_config.yaml'


def load_config(config_path=None) -> dict:
    p = Path(config_path) if config_path else _DEFAULT_CFG
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)
