import os

import torch

from src.image_branch.config import load_config
from src.image_branch.get_image import getImage
from src.image_branch.train import train
from src.image_branch.model import load_model
from src.image_branch.predictor import predict


def main():
    config = load_config()

    # Step 1: 下載圖片
    getImage(config)

    # Step 2-3: 訓練模型
    train(config)

    # Step 4: test set inference → parquet
    device = torch.device(
        config['training']['device']
        if torch.cuda.is_available()
        else 'cpu'
    )
    run_id   = config['output']['run_id']
    best_dir = os.path.join(config['output']['results_dir'], run_id, 'best')
    model    = load_model(config)
    model.load_state_dict(
        torch.load(os.path.join(best_dir, 'pytorch_model.bin'), map_location=device)
    )
    model.to(device)
    predict(model, config, device)


if __name__ == '__main__':
    main()
