from src.image_branch.config import load_config
from src.image_branch.train import train


def main():
    config = load_config()
    train(config)


if __name__ == '__main__':
    main()
