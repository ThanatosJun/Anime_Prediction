import yaml

def load_config(config_path='src/image_branch/configs/image_process_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
