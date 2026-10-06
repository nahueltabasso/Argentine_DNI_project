from app.services.pipeline import Pipeline
from app.services.readers.mrz_reader import MRZReader
from app.services.readers.pdf417_reader import PDF417Reader
from app.services.readers.address_reader import GeminiOCRAddressReader, PaddleOCRAddressReader
from app.services.document_detector import DocumentDetector
from app.services.ocr_service import OCRService
from app.services.elements_detector import ElementsDetector
from app.routers import document as document_router
from app.routers import health as health_router
from app.core.exception_handlers import business_exception_handler, unhandled_error_handler
from app.core.exceptions import BusinessLogicError
from app.core.settings import Settings, get_settings
from app.core.logging_config import setup_logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
     
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ocr = OCRService()
        address_reader = (
            GeminiOCRAddressReader(settings)
            if settings.address_strategy == "vlm"
            else PaddleOCRAddressReader(ocr)

        )
        if settings.load_models:
            app.state.document_detector = DocumentDetector(settings)
            app.state.pipeline = Pipeline(
                document_detector=DocumentDetector(settings),
                element_detector=ElementsDetector(settings),
                ocr_service=ocr,
                address_reader=address_reader,
                pdf417_reader=PDF417Reader(),
                mrz_reader=MRZReader(ocr),
            )
        yield

    app = FastAPI(title="Argentine DNI API", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(document_router.router, prefix=settings.api_prefix)
    app.include_router(health_router.router, prefix=settings.api_prefix)
    app.add_exception_handler(BusinessLogicError, business_exception_handler) # type: ignore
    app.add_exception_handler(Exception, unhandled_error_handler) # type: ignore
    return app
