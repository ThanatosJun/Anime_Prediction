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
    model    = load_model(config)
    model.load_state_dict(
        torch.load(os.path.join(best_dir, 'pytorch_model.bin'), map_location=device)
    )
    model.to(device)
    predict(model, config, device)


if __name__ == '__main__':
    main()
