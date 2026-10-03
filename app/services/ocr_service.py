from typing import Optional
from paddleocr import PaddleOCR
from google import genai
from google.genai.errors import APIError
from time import sleep
from PIL import Image
from app.core.settings import Settings
import threading
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OCRService:
   
    _instance: Optional["OCRService"] = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self._ocr = PaddleOCR(use_doc_orientation_classify=False,
                              use_doc_unwarping=False,
                              use_textline_orientation=False,
                              lang="en")
        self._client = genai.Client(
            api_key=settings.google_api_key.get_secret_value() if settings.google_api_key else None
        )
        self._prompt=settings.prompt
        
    @classmethod
    def initialize(cls, settings: Settings) -> "OCRService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(settings)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "OCRService":
        if cls._instance is None:
            raise RuntimeError("OCRService has not been initialized.")
        return cls._instance  
    
    def recognize_text(self, img: np.ndarray, join_char: str = ""):
        logger.info("Recognizing text from image using PaddleOCR.")
        result = self._ocr.predict(img)[0]
        text = result['rec_texts']
        return join_char.join(text)
    
    def recognize_text_with_gemini(self, image: Image.Image) -> str:
        logger.info("Recognizing text from image using Gemini.")
        processed = False
        while not processed:
            try:
                response = self._client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[self._prompt, image],
                )
                processed = True
            except APIError as e:
                if e.code == 429:
                    sleep(60)
                else:
                    raise e
        return response.text or ""