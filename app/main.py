from fastapi import FastAPI, status
from contextlib import asynccontextmanager
from app.core.exception_handlers import business_exception_handler, unhandled_error_handler
from app.core.exceptions import BusinessLogicError
from app.core.settings import Settings, get_settings
from app.core.logging_config import setup_logging
from app.services.document_detector import DocumentDetector
from app.services.elements_service import ElementsService
from app.services.ocr_service import OCRService
from app.routers import document as document_router
from app.routers import health as health_router
import logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
     
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.load_models:
            app.state.document_detector = DocumentDetector(settings)
            app.state.ocr_service = OCRService(settings)
            app.state.elements_service = ElementsService(
                settings, app.state.document_detector, app.state.ocr_service
            )
        yield

    app = FastAPI(title="Argentine DNI API", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(document_router.router, prefix=settings.api_prefix)
    app.include_router(health_router.router, prefix=settings.api_prefix)
    app.add_exception_handler(BusinessLogicError, business_exception_handler) # type: ignore
    app.add_exception_handler(Exception, unhandled_error_handler) # type: ignore
    return app


# @app.get("/health", status_code=status.HTTP_200_OK)
# def health_check():
#     return {"status": "ok"}