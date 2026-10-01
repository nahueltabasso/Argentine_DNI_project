from pathlib import Path
from typing import Optional
from datetime import datetime
from ultralytics import YOLO
from ultralytics.engine.results import Results
from app.core.settings import Settings
from app.core.exceptions import BusinessLogicError
from app.schemas.error_codes import ErrorCode
from app.schemas.schemas import ResponseData
import threading
import numpy as np
import logging

logger = logging.getLogger(__name__)

FRONT_CLASS = 0
BACK_CLASS = 1

class DocumentDetector:
    
    _instance: Optional["DocumentDetector"] = None
    _instance_lock = threading.Lock()
    
    def __init__(self, model_path: Path, settings: Settings) -> None:
        self._model = YOLO(str(model_path))
        self._conf_thresh = settings.doc_conf
        self._iou = settings.doc_iou
        self._img_size = settings.doc_imgsz
        self._device = settings.device
        
    @classmethod
    def initialize(cls, model_path: Path, settings: Settings) -> "DocumentDetector":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(model_path, settings)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "DocumentDetector":
        if cls._instance is None:
            raise RuntimeError("DocumentDetector has not been initialized.")
        return cls._instance
    
    def predict(self, image: np.ndarray) -> Results:
        return self._model.predict(
            source=image,
            conf=self._conf_thresh,
            iou=self._iou,
            imgsz=self._img_size,
            device=self._device
        )[0] # type: ignore
        
    def get_argentine_ID_card(self, image: np.ndarray) -> ResponseData:
        logger.info("Starting Argentine ID card detection.")
        result = self.predict(image)
        if result is None or len(result.obb) == 0: # type: ignore
            logger.error("No Argentine ID card detected.")
            raise BusinessLogicError("No Argentine ID card detected.", error_code=ErrorCode.INVALID_IMAGE_ERROR)
        if len(result.obb) > 1: # type: ignore
            logger.error("Multiple Argentine ID cards detected.")
            raise BusinessLogicError("Multiple Argentine ID cards detected.", error_code=ErrorCode.INVALID_IMAGE_ERROR)
        obb = result.obb
        print(f"OBB: {obb}")
        points = obb.xyxyxyxy[0].cpu().numpy().astype(int).tolist() # type: ignore
        cls = obb.cls[0].cpu().numpy().astype(int).tolist() # type: ignore
        print(f"Class: {cls}")
        if cls == FRONT_CLASS:
            response_data: ResponseData = ResponseData(
                front_coords=points,
                back_coords=None,
                timestamp=datetime.now(),
            ) # type: ignore
        elif cls == BACK_CLASS:
            response_data: ResponseData = ResponseData(
                front_coords=None,
                back_coords=points,
                timestamp=datetime.now(),
            ) # type: ignore
        else:
            logger.error("Unknown class detected.")
            raise BusinessLogicError("Unknown class detected.", error_code=ErrorCode.INVALID_IMAGE_ERROR)

        return response_data
