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


# ── aug helpers ───────────────────────────────────────────────────────────────

def _make_crop(config) -> T.RandomResizedCrop:
    cfg = config['augmentation']['random_resized_crop']
    return T.RandomResizedCrop(224, scale=cfg['scale'], ratio=cfg['ratio'])


def _make_random_crop(config) -> T.RandomApply:
    cfg = config['augmentation']['random_crop']
    padding = int(224 * cfg['max_crop_ratio'])
    return T.RandomApply([T.RandomCrop(224, padding=padding)], p=cfg['p'])


def _make_color_jitter(config) -> T.RandomApply:
    cfg = config['augmentation']['color_jitter']
    return T.RandomApply([
        T.ColorJitter(
            brightness=cfg['brightness'],
            contrast=cfg['contrast'],
            saturation=cfg['saturation'],
            hue=cfg['hue'],
        )
    ], p=cfg['p'])


def _make_gaussian_blur(config) -> T.RandomApply:
    cfg = config['augmentation']['gaussian_blur']
    return T.RandomApply([
        T.GaussianBlur(kernel_size=cfg['kernel_size'], sigma=cfg['sigma'])
    ], p=cfg['p'])


def _make_flip(config) -> T.RandomHorizontalFlip:
    p = config['augmentation']['random_horizontal_flip']['p']
    return T.RandomHorizontalFlip(p=p)


def _make_grayscale(config) -> T.RandomGrayscale:
    p = config['augmentation']['random_grayscale']['p']
    return T.RandomGrayscale(p=p)


def get_transform_aug(config) -> T.Compose:
    return T.Compose([
        _make_crop(config),
        _make_random_crop(config),
        _make_color_jitter(config),
        _make_gaussian_blur(config),
        _make_flip(config),
        _make_grayscale(config),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
