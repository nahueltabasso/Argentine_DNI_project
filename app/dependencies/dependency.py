from fastapi import Depends, Request
from typing import Annotated
from app.services.document_detector import DocumentDetector
from app.services.elements_service import ElementsService
from app.services.ocr_service import OCRService

def get_document_detector(request: Request) -> DocumentDetector:
    return request.app.state.document_detector

def get_ocr_service(request: Request) -> OCRService:
    return request.app.state.ocr_service

def get_elements_service(request: Request) -> ElementsService:
    return request.app.state.elements_service

DocumentDetectorDep = Annotated[DocumentDetector, Depends(get_document_detector)]
OCRServiceDep = Annotated[OCRService, Depends(get_ocr_service)]
ElementsServiceDep = Annotated[ElementsService, Depends(get_elements_service)]
