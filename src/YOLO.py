from typing import List, Optional, Tuple

from imgutils.detect import detect_person as _imgutils_detect_person
from PIL import Image


def detect_person(
    image: Image.Image,
    level: str = "m",
    version: str = "v1.1",
    model_name: Optional[str] = None,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> List:
    kwargs = {}
    if model_name is not None:
        kwargs["model_name"] = model_name
    # ((x0, y0, x1, y1), 'person', confidence_score)
    return _imgutils_detect_person(
        image,
        level=level,
        version=version,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
