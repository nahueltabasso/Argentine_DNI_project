import logging

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.exceptions import BusinessLogicError
from app.schemas.error_codes import ErrorCode

logger = logging.getLogger(__name__)


def image_to_ndarray(file: UploadFile) -> np.ndarray:
    """Convert an uploaded image file to a Numpy ndarray."""
    file_bytes = file.file.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise BusinessLogicError(
            "Failed to decode image.", error_code=ErrorCode.INVALID_IMAGE_ERROR
        )
    return image


def order_points(points: list[list[int]]) -> np.ndarray:
    """Order points in the following order: top-left, top-right, bottom-right, bottom-left."""
    points_array = np.array(points, dtype=np.float32)

    s = points_array.sum(axis=1)
    diff = np.diff(points_array, axis=1).flatten()

    top_left = points_array[np.argmin(s)]
    top_right = points_array[np.argmin(diff)]
    bottom_right = points_array[np.argmax(s)]
    bottom_left = points_array[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def rectify_obb(
    image: np.ndarray, obb_points: list[list[int]], target_size=(1200, 756)
) -> np.ndarray:
    """Rectify an oriented bounding box (OBB) in an image to a horizontal rectangle."""
    src = order_points(obb_points)

    target_width, target_height = target_size

    # Width and height of the detected document
    width = np.linalg.norm(src[1] - src[0])
    height = np.linalg.norm(src[3] - src[0])

    # --------------------------------------------------
    # If the document is vertical, rotate the points to make it horizontal
    # --------------------------------------------------
    if height > width:
        # Rotate the order of the points
        src = np.array(
            [
                src[3],  # top-left
                src[0],  # top-right
                src[1],  # bottom-right
                src[2],  # bottom-left
            ],
            dtype=np.float32,
        )

    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    # Homography
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Transformation
    return cv2.warpPerspective(image, matrix, (target_width, target_height))
