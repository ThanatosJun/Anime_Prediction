from src.config import load_config
from util.train import train


def main():
    config = load_config()
    train(config)


if __name__ == '__main__':
    main()
