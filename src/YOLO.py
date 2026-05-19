from typing import List, Optional, Tuple

from imgutils.detect import detect_person as _imgutils_detect_person
from imgutils.detect import detect_faces as _imgutils_detect_faces
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


def detect_faces(
    image: Image.Image,
    level: str = "s",
    version: str = "v1.4",
    model_name: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    # ((x0, y0, x1, y1), 'face', confidence_score)
    kwargs = {}
    if model_name is not None:
        kwargs["model_name"] = model_name
    return _imgutils_detect_faces(
        image,
        level=level,
        version=version,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        **kwargs,
    )
