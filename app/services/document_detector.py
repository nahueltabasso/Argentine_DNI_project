import logging
from datetime import UTC, datetime

import numpy as np
from ultralytics import YOLO

from app.core.exceptions import BusinessLogicError
from app.core.settings import Settings
from app.schemas.error_codes import ErrorCode
from app.schemas.schemas import DocumentDetected, SidesName
from app.services.inference import Inference

logger = logging.getLogger(__name__)

FRONT_CLASS = 0
BACK_CLASS = 1


class DocumentDetector(Inference):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            model=YOLO(str(settings.yolo_dni_detector)),
            conf_thresh=settings.doc_conf,
            iou=settings.doc_iou,
            img_size=settings.doc_imgsz,
            device=settings.device,
        )

    def get_argentine_ID_card(self, image: np.ndarray) -> DocumentDetected:
        logger.info("Starting Argentine ID card detection.")
        result = self.predict(image)

        if result.obb is None or len(result.obb) == 0:
            logger.error("No Argentine ID card detected.")
            raise BusinessLogicError(
                "No Argentine ID card detected.",
                error_code=ErrorCode.DOCUMENT_NOT_FOUND_ERROR,
            )
        if len(result.obb) > 1:
            logger.error("Multiple Argentine ID cards detected.")
            raise BusinessLogicError(
                "Multiple Argentine ID cards detected.",
                error_code=ErrorCode.MULTIPLE_DOCUMENTS_ERROR,
            )

        obb = result.obb
        points = obb.xyxyxyxy[0].cpu().numpy().astype(int).tolist()
        cls_ = int(obb.cls[0])
        conf = float(obb.conf[0])

        if cls_ not in [FRONT_CLASS, BACK_CLASS]:
            logger.error("Unknown class detected.")
            raise BusinessLogicError(
                "Unknown class detected.", error_code=ErrorCode.UNKNOWN_CLASS_ERROR
            )
        return DocumentDetected(
            side=SidesName.FRONT if cls_ == FRONT_CLASS else SidesName.BACK,
            points=points,
            confidence=round(conf, 2),
            timestamp=datetime.now(UTC),
        )
