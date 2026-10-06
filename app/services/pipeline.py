
from app.core.exceptions import BusinessLogicError
from app.schemas.error_codes import ErrorCode
from app.schemas.schemas import ArgentineIDData, DocumentDetected, ElementDetection, ElementsID
from app.services.readers.pdf417_reader import PDF417Reader
from app.services.readers.mrz_reader import MRZReader
from app.services.readers.address_reader import AddressReader
from app.services.document_detector import DocumentDetector
from app.services.elements_detector import ElementsDetector
from app.services.ocr_service import OCRService
from app.utils.file_utils import rectify_obb
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)
GENDERS = {"F": "Female", "M": "Male", "X": "X"}

class Pipeline:

    def __init__(self,
                 document_detector: DocumentDetector,
                 element_detector: ElementsDetector,
                 ocr_service: OCRService,
                 pdf417_reader: PDF417Reader,
                 mrz_reader: MRZReader,
                 address_reader: AddressReader) -> None:
        self._doc_detector = document_detector
        self._element_detector = element_detector
        self._ocr_service = ocr_service
        self._pdf417_reader = pdf417_reader
        self._mrz_reader = mrz_reader
        self._address_reader = address_reader

    def extract_full(self,
                     front_image: np.ndarray,
                     back_image: np.ndarray) -> ArgentineIDData:
        """Extract data from both the front and back images of an Argentine ID card."""
        logger.info("Enter to extract_full()")
        logger.info("Detecting document in front image %s", front_image.shape)
        front_side: DocumentDetected = self._doc_detector.get_argentine_ID_card(image=front_image)
        logger.info("Detecting document in back image %s", back_image.shape)
        back_side: DocumentDetected = self._doc_detector.get_argentine_ID_card(image=back_image)
  
        if front_side.side == back_side.side:
            raise BusinessLogicError(message="Front and back images are the same side.",
                                     error_code=ErrorCode.SAME_SIDE_ERROR)

        logger.info("Processing document image for element extraction.")
        logger.info("Cropping and rectifying the document image.")
        front_card = rectify_obb(image=front_image, obb_points=front_side.points)
        back_card = rectify_obb(image=back_image, obb_points=back_side.points)
        images = [front_card, back_card]
        # Detect elements in the cropped and rectified images.
        elements: dict[ElementsID, ElementDetection] = self._element_detector.detect_elements(images=images)
        logger.info("Detected %d elements", len(elements))
        return self._build_data(elements=elements)

    def extract_side(self,
                     image: np.ndarray,
                     side: str = "") -> ArgentineIDData:
        """Extract data from a specific side of an Argentine ID card."""
        logger.info("Extracting side %s from image", side)
        doc_rec = self._doc_detector.get_argentine_ID_card(image=image)
        if doc_rec.side != side:
            raise BusinessLogicError("The side of the detected Argentine ID not match with the requested side.",
                                     error_code=ErrorCode.NOT_MATCH_SIDE_ERROR)
        img = rectify_obb(image=image, obb_points=doc_rec.points)
        elements: dict[ElementsID, ElementDetection] = self._element_detector.detect_elements(images=[img])
        logger.info("Detected %d elements for the %s side", len(elements), side)
        return self._build_data(elements=elements, side=side)

    def _build_data(self,
                    elements: dict[ElementsID, ElementDetection],
                    side: str = "both") -> ArgentineIDData:
        logger.info("Building ArgentineIDData with elements: %s", list(elements.keys()))
        return ArgentineIDData(
            side=side,
            doc_number=self._recognize_element(elements.get(ElementsID.DOC_NUMBER_CLS)),
            tramite_number=self._recognize_element(elements.get(ElementsID.TRAMITE_NUMBER_CLS)),
            has_shield=ElementsID.SHIELD_CLS in elements,
            has_picture=ElementsID.PICTURE_CLS in elements,
            has_country=ElementsID.COUNTRY_CLS in elements,
            pdf417=self._pdf417_reader.read(
                element=elements.get(ElementsID.PDF417_CLS)
            ),
            mrz=self._mrz_reader.read(
                element=elements.get(ElementsID.MRZ_CLS)
            ),
            address=self._address_reader.read(
                element=elements.get(ElementsID.ADDRESS_CLS)
            ),
            gender=GENDERS.get(self._recognize_element(elements.get(ElementsID.GENDER_CLS))),
            timestamp=datetime.now()
        )

    def _recognize_element(self,
                           element: ElementDetection | None,
                           join_char: str = "") -> str | None:
        """Run OCR over a detected element, or return None if it was not detected."""
        if element is None:
            return None
        return self._ocr_service.recognize_text(img=element.crop, join_char=join_char)