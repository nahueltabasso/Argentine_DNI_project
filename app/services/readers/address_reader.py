import logging
from time import sleep
from typing import Protocol

import cv2
from google import genai
from google.genai.errors import APIError
from PIL import Image

from app.core.settings import Settings
from app.schemas.schemas import ElementDetection
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)


class AddressReader(Protocol):
    def read(
        self, element: ElementDetection | None, join_char: str = " "
    ) -> str | None: ...


class PaddleOCRAddressReader:
    def __init__(self, ocr_service: OCRService) -> None:
        self._ocr_service = ocr_service

    def read(
        self, element: ElementDetection | None, join_char: str = " "
    ) -> str | None:
        if element is None:
            return None
        return self._ocr_service.recognize_text(img=element.crop, join_char=join_char)


class GeminiOCRAddressReader:
    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(api_key=settings.google_api_key.get_secret_value())
        self._model = settings.vlm_model
        self._prompt = settings.prompt
        self._max_retries = settings.vlm_max_retries
        self._timeout = settings.vlm_timeout_s

    def read(
        self, element: ElementDetection | None, join_char: str = " "
    ) -> str | None:
        if element is None:
            return None
        logger.info("Recognizing text from image using Gemini.")
        image: Image.Image = Image.fromarray(
            cv2.cvtColor(element.crop, cv2.COLOR_BGR2RGB)
        )
        for _ in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[self._prompt, image],
                )
                return response.text or None
            except APIError as e:
                if e.code == 429:
                    sleep(self._timeout)
                else:
                    raise e
        return None
