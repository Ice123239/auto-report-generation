"""Image preprocessing helpers used before inference."""

from __future__ import annotations

import cv2
import numpy as np


MAX_INFERENCE_IMAGE_WIDTH = 1024


def resize_image_width(
    image: np.ndarray,
    max_width: int = MAX_INFERENCE_IMAGE_WIDTH,
) -> np.ndarray:
    """Resize an image to a maximum width while preserving its aspect ratio."""
    if image.shape[1] <= max_width:
        return image

    height = max(1, round(image.shape[0] * max_width / image.shape[1]))
    return cv2.resize(
        image,
        (max_width, height),
        interpolation=cv2.INTER_AREA,
    )
