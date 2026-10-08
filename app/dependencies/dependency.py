from typing import Annotated

from fastapi import Depends, Request

from app.services.document_detector import DocumentDetector
from app.services.pipeline import Pipeline


def get_document_detector(request: Request) -> DocumentDetector:
    return request.app.state.document_detector


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


DocumentDetectorDep = Annotated[DocumentDetector, Depends(get_document_detector)]
PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
