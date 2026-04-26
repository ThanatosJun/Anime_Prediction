import torch

from src.config import load_config
from util.getImage import getImage
from util.train import train
from util.predictor import predict


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
    from src.model import load_model
    import os
    run_id   = config['output']['run_id']
    best_dir = os.path.join(config['output']['results_dir'], run_id, 'best')
    from src.model import load_model
    from transformers import SwinModel
    model    = SwinModel.from_pretrained(best_dir).to(device)
    predict(model, config, device)


if __name__ == '__main__':
    main()
