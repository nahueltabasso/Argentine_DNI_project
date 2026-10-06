from app.services.document_detector import DocumentDetector
from app.services.elements_detector import ElementsDetector
from app.services.pipeline import Pipeline
from app.services.ocr_service import OCRService
from fastapi import Depends, Request
from typing import Annotated

def get_document_detector(request: Request) -> DocumentDetector:
    return request.app.state.document_detector

def get_ocr_service(request: Request) -> OCRService:
    return request.app.state.ocr_service

def get_elements_detector(request: Request) -> ElementsDetector:
    return request.app.state.elements_detector

def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline

DocumentDetectorDep = Annotated[DocumentDetector, Depends(get_document_detector)]
OCRServiceDep = Annotated[OCRService, Depends(get_ocr_service)]
ElementsDetectorDep = Annotated[ElementsDetector, Depends(get_elements_detector)]
PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
