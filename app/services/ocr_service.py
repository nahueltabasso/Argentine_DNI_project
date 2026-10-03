from typing import Optional
from paddleocr import PaddleOCR
import threading
import numpy as np

class OCRService:
    
    _instance: Optional["OCRService"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._ocr = PaddleOCR(use_doc_orientation_classify=False,
                              use_doc_unwarping=False,
                              use_textline_orientation=False,
                              lang="en")
        
    @classmethod
    def initialize(cls) -> "OCRService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "OCRService":
        if cls._instance is None:
            raise RuntimeError("OCRService has not been initialized.")
        return cls._instance  
    
    def recognize_text(self, img: np.ndarray, join_char: str = ""):
        result = self._ocr.predict(img)[0]
        text = result['rec_texts']
        return join_char.join(text)