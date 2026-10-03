from fastapi import FastAPI, status
from contextlib import asynccontextmanager
from app.core.exception_handlers import business_exception_handler, unhandled_error_handler
from app.core.exceptions import BusinessLogicError
from app.core.settings import get_settings
from app.core.logging_config import setup_logging
from app.services.document_detector import DocumentDetector
from app.services.elements_service import ElementsService
from app.services.ocr_service import OCRService
from app.routers import document as document_router
import logging

logger = logging.getLogger(__name__)
settings = get_settings()    

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Loading models...")
    app.state.document_detector = DocumentDetector(settings)
    app.state.ocr_service = OCRService(settings)
    app.state.elements_service = ElementsService(
        settings,
        app.state.document_detector,
        app.state.ocr_service,
    )
    logger.info("Models loaded")

    yield    
    app.state.clear()


app = FastAPI(lifespan=lifespan)
# Include Routers
app.include_router(document_router.router, prefix=settings.api_prefix)

# Include exception handlers
app.add_exception_handler(BusinessLogicError, business_exception_handler) # type: ignore
app.add_exception_handler(Exception, unhandled_error_handler) # type: ignore

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok"}