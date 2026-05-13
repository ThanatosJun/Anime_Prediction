from src.config import load_config
from util.getImage import getImage


def main():
    config = load_config()
    getImage(config)


if __name__ == '__main__':
    main()
