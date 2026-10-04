from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import threading

class Inference:
    
    def __init__(self, model: YOLO,
                 conf_thresh: float,
                 iou: float,
                 img_size: int,
                 device: str) -> None:
        self._model = model
        self._conf_thresh = conf_thresh
        self._iou = iou
        self._img_size = img_size
        self._device = device
        self._lock = threading.Lock()

    def predict(self, image: np.ndarray) -> Results:
        with self._lock:
            return self._model.predict(
                source=image,
                conf=self._conf_thresh,
                iou=self._iou,
                imgsz=self._img_size,
                device=self._device
            )[0] # type: ignore