from ultralytics import YOLO
from typing import Optional
from datetime import datetime
from PIL import Image
from mrz.checker.td1 import TD1CodeChecker
from itertools import zip_longest
from app.services.document_detector import DocumentDetector
from app.core.exceptions import BusinessLogicError
from app.schemas.error_codes import ErrorCode
from app.schemas.schemas import ArgentineIDData, DocumentDetected, ElementDetection, ElementsID, MRZData
from app.services.inference import Inference
from app.core.settings import Settings
from app.services.ocr_service import OCRService
from app.utils.file_utils import rectify_obb
import threading
import numpy as np
import logging
import zxing

logger = logging.getLogger(__name__)
PDF417_FIELDS = (
    "tramite_number", "surname", "name", "gender",
    "document_number", "category", "birth_date", "issue_date",
)

class ElementsService(Inference):
    _instance: Optional["ElementsService"] = None
    _instance_lock = threading.Lock()

    def __init__(self, 
                 settings: Settings, 
                 doc_detector: DocumentDetector,
                 ocr_service: OCRService) -> None:
        self._settings = settings
        self._doc_detector = doc_detector
        self._ocr_service = ocr_service
        self._pdf417_reader = zxing.BarCodeReader() 
        super().__init__(
            model=YOLO(str(settings.yolo_id_elements_detector)), 
            conf_thresh=settings.elem_conf,
            iou=settings.elem_iou,
            img_size=settings.elem_imgsz,
            device=settings.device
        )
        
    @classmethod
    def initialize(cls, settings: Settings) -> "ElementsService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(settings,
                                    DocumentDetector.get_instance(), 
                                    OCRService.get_instance())
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "ElementsService":
        if cls._instance is None:
            raise RuntimeError("ElementsService has not been initialized.")
        return cls._instance
    
    def get_data_from_doc(self, 
                          front_image: np.ndarray,
                          back_image: np.ndarray) -> ArgentineIDData | None:
        """Extract Data from Argentine ID Card."""
        logger.info("Enter to get_data_from_doc()")
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
        elements: dict[ElementsID, ElementDetection] = self._detect_elements(images=images)
        logger.info("Detected %d elements", len(elements))
        return self._build_data(elements=elements)
    
    def _detect_elements(self, images: list) -> dict[ElementsID, ElementDetection]:
        """Detect elements of Argentine ID Card from images and return a dictionary mapping ElementsID to ElementDetection."""
        elements: dict[ElementsID, ElementDetection] = {}
        for img in images:
            result = self.predict(image=img)
            
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = [list(map(int, box)) for box in result.boxes.xyxy.tolist()]
            classes = [int(cls) for cls in result.boxes.cls.tolist()]
            confs = [float(conf) for conf in result.boxes.conf.tolist()]
            for box, cls_, conf in zip(xyxy, classes, confs):
                element = ElementsID(int(cls_))
                if element not in elements:
                    elements[element] = ElementDetection(
                        box=box,
                        conf=conf,
                        crop=img[box[1]:box[3], box[0]:box[2]]
                    )
        return elements
    
    def _build_data(self,
                    elements: dict[ElementsID, ElementDetection],
                    side: str = "") -> ArgentineIDData | None:
        return ArgentineIDData(
            side=side,
            doc_number=self._read_text(element=elements.get(ElementsID.DOC_NUMBER_CLS), join_char=""),
            tramite_number=self._read_text(element=elements.get(ElementsID.TRAMITE_NUMBER_CLS), join_char=""), 
            has_shield=ElementsID.SHIELD_CLS in elements,
            has_picture=ElementsID.PICTURE_CLS in elements,
            has_country=ElementsID.COUNTRY_CLS in elements,
            gender=self._get_gender(element=elements.get(ElementsID.GENDER_CLS), join_char=""),
            pdf417=self._read_pdf417(element=elements.get(ElementsID.PDF417_CLS)),
            mrz=self._read_mrz(element=elements.get(ElementsID.MRZ_CLS)), 
            address=self._read_address(element=elements.get(ElementsID.ADDRESS_CLS)),
            timestamp=datetime.now()
        ) # type: ignore
        
    def _read_text(self,
                   element: ElementDetection | None,
                   join_char: str = "") -> str | None:
        """Read text from the specified element in the image."""
        if element is None:
            return None
        return self._ocr_service.recognize_text(element.crop, join_char=join_char)
    
    def _get_gender(self, 
                    element: ElementDetection | None,
                    join_char: str = "") -> str | None:
        """Return a gender from the specified element in the image."""
        if element is None:
            return None
        logger.info("Getting gender from element in image of shape: %s", element.crop.shape)
        text = self._read_text(element=element, join_char=join_char)
        text = "Female" if text == 'F' else "Male" if text == 'M' else None
        logger.info("Detected gender: %s", text)
        return text
    
    def _read_pdf417(self, element: ElementDetection | None) -> dict | None:
        """Read PDF417 barcode from the specified element in the image using zxing."""
        if element is None:
            return None
        logger.info("Reading PDF417 from element in image of shape: %s", element.crop.shape)
        cropped_img = Image.fromarray(element.crop)
        barcode = self._pdf417_reader.decode(cropped_img)
        raw = barcode.raw if barcode is not None else None
        if raw is not None:
            raw = raw.split("@")
            return dict(zip_longest(PDF417_FIELDS, raw[:len(PDF417_FIELDS)]))
        logger.warning(
            "Failed to read PDF417 from element in image of shape: %s", element.crop.shape
        )
        return None
        
    def _read_mrz(self,
                  element: ElementDetection | None,
                  join_char: str = "\n") -> MRZData | None:
        """Read MRZ (Machine readable zone) from the specified element in the image."""
        if element is None:
            return None
        logger.info("Reading MRZ from element in image of shape: %s", element.crop.shape)
        mrz_text = self._read_text(element=element, join_char=join_char)
        
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
    
    def _read_address(self,
                      element: ElementDetection | None) -> str | None:
        """Read address from the specified element in the image."""
        logger.info("Reading address from element in image.")
        if self._settings.address_strategy != "vlm":
            return self._read_text(element=element, join_char=" ")
        return self._ocr_service.recognize_text_with_gemini(Image.fromarray(element.crop)) # type: ignore
    
    
    def get_data_from_side(self, image: np.ndarray, side: str = "") -> ArgentineIDData | None:
        """Extract data from the specified side of the Argentine ID."""
        logger.info("Extracting data from side: %s", side)
        doc_rec = self._doc_detector.get_argentine_ID_card(image=image)
        if doc_rec.side != side:
            raise BusinessLogicError("The side of the detected Argentine ID not match with the requested side.",
                                     error_code=ErrorCode.NOT_MATCH_SIDE_ERROR)
        img = rectify_obb(image=image, obb_points=doc_rec.points)
        elements: dict[ElementsID, ElementDetection] = self._detect_elements(images=[img])
        logger.info("Detected %d elements for the %s side", len(elements), side)
        return self._build_data(elements=elements, side=side)