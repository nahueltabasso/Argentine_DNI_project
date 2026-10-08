import logging

from ultralytics import YOLO

from app.core.settings import Settings
from app.schemas.schemas import ElementDetection, ElementsID
from app.services.inference import Inference

logger = logging.getLogger(__name__)


class ElementsDetector(Inference):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            model=YOLO(str(settings.yolo_id_elements_detector)),
            conf_thresh=settings.elem_conf,
            iou=settings.elem_iou,
            img_size=settings.elem_imgsz,
            device=settings.device,
        )

    def detect_elements(self, images: list) -> dict[ElementsID, ElementDetection]:
        """Detect elements of Argentine ID Card from images and return a dictionary mapping ElementsID to ElementDetection."""
        elements: dict[ElementsID, ElementDetection] = {}
        for img in images:
            result = self.predict(image=img)

            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = [list(map(int, box)) for box in result.boxes.xyxy.tolist()]
            classes = [int(cls) for cls in result.boxes.cls.tolist()]
            confs = [float(conf) for conf in result.boxes.conf.tolist()]
            for box, cls_, conf in zip(xyxy, classes, confs, strict=True):
                element = ElementsID(int(cls_))
                if element not in elements:
                    elements[element] = ElementDetection(
                        box=box, conf=conf, crop=img[box[1] : box[3], box[0] : box[2]]
                    )
        return elements
