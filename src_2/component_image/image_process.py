from PIL import Image, ImageOps
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ResizeWithPad:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left   = pad_w // 2
        right  = pad_w - left
        top    = pad_h // 2
        bottom = pad_h - top

        return ImageOps.expand(img, border=(left, top, right, bottom), fill=0)


def load_image(path: str):
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return None


def get_transform_original(image_size: int) -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
