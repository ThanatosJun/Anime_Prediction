import yaml


def load_config(config_path='image_encoder_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_yolo_config(config_path='image_encoder_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        full = yaml.safe_load(f)
    return full.get('yolo_detection', {})
