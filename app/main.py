from fastapi import FastAPI
from app.core.exception_handlers import business_exception_handler
from app.core.exceptions import BusinessLogicError
from app.core.settings import get_settings
from app.core.logging_config import setup_logging
import logging

app = FastAPI()
app.add_exception_handler(BusinessLogicError, business_exception_handler) # type: ignore

settings = get_settings()    
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)
    
@app.get("/health", status_code=200)
def health_check():
    logger.info("Health check endpoint called")
    logger.debug("Google API Key: %s", settings.google_api_key.get_secret_value()) # type: ignore
    print(settings.google_api_key.get_secret_value()) # type: ignore
    print(settings.yolo_dni_detector)
    raise BusinessLogicError("Health check failed", error_code="TEST")
    return {"status": "ok"}