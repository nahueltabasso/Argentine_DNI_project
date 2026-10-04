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
   
    def __init__(self, settings: Settings) -> None:
        self._ocr = PaddleOCR(use_doc_orientation_classify=False,
                              use_doc_unwarping=False,
                              use_textline_orientation=False,
                              lang="en")
        self._paddle_lock = threading.Lock()
        if settings.address_strategy == "vlm":
            self._client = genai.Client(
                api_key=settings.google_api_key.get_secret_value()
            )
        self._settings = settings
        self._prompt=settings.prompt
        
    def recognize_text(self, img: np.ndarray, join_char: str = ""):
        logger.info("Recognizing text from image using PaddleOCR.")
        with self._paddle_lock:
            result = self._ocr.predict(img)[0]
        text = result['rec_texts']
        return join_char.join(text)
    
    def recognize_text_with_gemini(self, image: Image.Image) -> str:
        logger.info("Recognizing text from image using Gemini.")
        for _ in range(self._settings.vlm_max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._settings.vlm_model,
                    contents=[self._prompt, image],
                )
                return response.text or ""
            except APIError as e:
                if e.code == 429:
                    sleep(self._settings.vlm_timeout_s)
                else:
                    raise e
        return ""