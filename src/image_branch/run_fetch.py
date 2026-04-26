from src.image_branch.config import load_config
from src.image_branch.get_image import getImage


def main():
    config = load_config()
    getImage(config)


if __name__ == '__main__':
    main()
