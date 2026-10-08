import logging
import threading

import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self) -> None:
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
        self._paddle_lock = threading.Lock()

    def recognize_text(self, img: np.ndarray, join_char: str = ""):
        logger.info("Recognizing text from image using PaddleOCR.")
        with self._paddle_lock:
            result = self._ocr.predict(img)[0]
        text = result["rec_texts"]
        return join_char.join(text)
