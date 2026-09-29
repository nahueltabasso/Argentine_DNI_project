from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.exceptions import BusinessLogicError
import logging

logger = logging.getLogger(__name__)

async def business_exception_handler(
    request: Request, 
    exc: BusinessLogicError) -> JSONResponse:
    logger.warning(
        "%s en %s %s: %s",
        exc.error_code, request.method, request.url.path, exc.message,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": status.HTTP_400_BAD_REQUEST,
            "message": exc.message,
            "method": request.method,
            "url": str(request.url),
        }
    )    

async def unhandled_error_handler(
    request: Request,
    exc: Exception) -> JSONResponse:
    logger.exception("Error not handled in %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error_code": "INTERNAL_ERROR", "message": "Internal server error"},
    )
