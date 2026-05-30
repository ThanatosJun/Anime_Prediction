import yaml
from pathlib import Path

_DEFAULT_CFG = Path(__file__).parent / 'image_encoder_config.yaml'


def load_config(config_path=None):
    p = Path(config_path) if config_path else _DEFAULT_CFG
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_yolo_config(config_path=None):
    p = Path(config_path) if config_path else _DEFAULT_CFG
    with open(p, 'r', encoding='utf-8') as f:
        full = yaml.safe_load(f)
    return full.get('yolo_detection', {})
