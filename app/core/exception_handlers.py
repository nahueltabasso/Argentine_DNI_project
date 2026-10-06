from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.exceptions import BusinessLogicError
from app.schemas.error_codes import ErrorCode
import logging

logger = logging.getLogger(__name__)

ERROR_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_IMAGE_ERROR: status.HTTP_400_BAD_REQUEST,
    ErrorCode.DOCUMENT_NOT_FOUND_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ErrorCode.MULTIPLE_DOCUMENTS_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ErrorCode.SAME_SIDE_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ErrorCode.NOT_MATCH_SIDE_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
    ErrorCode.UNKNOWN_CLASS_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


async def business_exception_handler(
    request: Request, 
    exc: BusinessLogicError) -> JSONResponse:
    logger.warning(
        "%s en %s %s: %s",
        exc.error_code, request.method, request.url.path, exc.message,
    )
    return JSONResponse(
        status_code=ERROR_STATUS.get(exc.error_code, status.HTTP_400_BAD_REQUEST),
        content={
            "status": ERROR_STATUS.get(exc.error_code, status.HTTP_400_BAD_REQUEST),
            "message": exc.message,
            "error_code": exc.error_code,
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
        content={
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR", 
            "method": request.method,
            "url": str(request.url),
        }
    )
