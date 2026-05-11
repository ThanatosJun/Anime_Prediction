import os

import torch
from transformers import SwinModel

from src.config import load_config, load_yolo_config
from util.predictor import predict


def main():
    config = load_config()

    device = torch.device(
        config['training']['device'] if torch.cuda.is_available() else 'cpu'
    )
    run_id   = config['output']['run_id']
    best_dir = os.path.join(config['output']['results_dir'], run_id, 'best')
    model    = SwinModel.from_pretrained(best_dir).to(device)
    yolo_config = load_yolo_config()
    use_yolo = yolo_config.get('yolo', {}).get('use', False)

    predict(model, config, device,use_yolo)


if __name__ == '__main__':
    main()
