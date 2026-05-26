from typing import List

from PIL import Image


def detect_and_crop(
    img: Image.Image,
    level: str = "m",
    version: str = "v1.1",
    conf_threshold: float = 0.2,
    iou_threshold: float = 0.8,
    max_persons: int = 5,
    fallback_full_image: bool = True,
    **kwargs,
) -> List[Image.Image]:
    """Detect anime characters and return cropped regions.

    Upscales small images before detection to improve recall.
    Detection and cropping both operate in the upscaled image space —
    no coordinate conversion needed because bbox coords are relative to img_det.
    Falls back to [img_det] when no person is detected.

    Returns a non-empty list of PIL Images (when fallback_full_image=True).
    """
    from imgutils.detect import detect_person as _detect

    w, h = img.size
    scale = max(640 / w, 640 / h)
    if scale > 1:
        img_det = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        img_det = img

    results = _detect(
        img_det,
        level=level,
        version=version,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    # Sort by confidence descending, take top-N
    results = sorted(results, key=lambda x: x[2], reverse=True)[:max_persons]

    crops: List[Image.Image] = []
    for bbox, _label, _conf in results:
        x0, y0, x1, y1 = bbox
        crops.append(img_det.crop((x0, y0, x1, y1)))  # bbox coords in img_det space

    if not crops:
        if fallback_full_image:
            return [img_det]
        return []

    return crops
