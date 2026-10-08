import logging

from mrz.checker.td1 import TD1CodeChecker

from app.schemas.schemas import ElementDetection, MRZData
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)


class MRZReader:
    def __init__(self, ocr_service: OCRService) -> None:
        self._ocr_service = ocr_service

    def read(
        self, element: ElementDetection | None, join_char: str = "\n"
    ) -> MRZData | None:
        """Read MRZ (Machine readable zone) from the specified element in the image."""
        if element is None:
            return None
        logger.info(
            "Reading MRZ from element in image of shape: %s", element.crop.shape
        )
        mrz_text = self._ocr_service.recognize_text(
            img=element.crop, join_char=join_char
        )

        try:
            check = TD1CodeChecker(mrz_text) if mrz_text else ""
            result = bool(check)
            if result:
                mrz_data: MRZData = MRZData.model_validate(check.fields())
                mrz_data.is_valid = result
                return mrz_data
            return None
        except Exception as e:
            logger.error("Failed to read MRZ: %s", e)
            return None
