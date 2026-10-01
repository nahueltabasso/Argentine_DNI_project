from fastapi import UploadFile
import numpy as np
import cv2

def image_to_ndarray(file: UploadFile) -> np.ndarray:
    file_bytes = file.file.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR) # type: ignore