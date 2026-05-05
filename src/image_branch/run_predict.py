import torch
from transformers import SwinModel
import os

from src.image_branch.config import load_config
from src.image_branch.predictor import predict


def main():
    config = load_config()
    device = torch.device(
        config['training']['device'] if torch.cuda.is_available() else 'cpu'
    )
    run_id   = config['output']['run_id']
    best_dir = os.path.join(config['output']['results_dir'], run_id, 'best')

    print(f"Loading model from {best_dir}…")
    model = SwinModel.from_pretrained(best_dir).to(device)

    predict(model, config, device)


if __name__ == '__main__':
    main()
