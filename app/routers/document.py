import logging
from typing import Annotated, Literal

from fastapi import APIRouter, File, UploadFile, status

from app.dependencies.dependency import DocumentDetectorDep, PipelineDep
from app.schemas.schemas import ArgentineIDData, DocumentDetected
from app.utils.file_utils import image_to_ndarray

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/document/detect", status_code=status.HTTP_200_OK, response_model=DocumentDetected
)
def detect_document(detector: DocumentDetectorDep, file: Annotated[UploadFile, File()]):
    logger.info("Enter to detect_document endpoint")
    image = image_to_ndarray(file)
    return detector.get_argentine_ID_card(image)


@router.post(
    "/document/extract-full",
    status_code=status.HTTP_200_OK,
    response_model=ArgentineIDData,
)
def extract_full_document(
    pipeline: PipelineDep,
    front: Annotated[UploadFile, File()],
    back: Annotated[UploadFile, File()],
):
    logger.info("Enter to extract_full_document endpoint")
    front_img = image_to_ndarray(front)
    back_img = image_to_ndarray(back)
    return pipeline.extract_full(front_img, back_img)


@router.post(
    "/document/extract", status_code=status.HTTP_200_OK, response_model=ArgentineIDData
)
def extract(
    side: Literal["front", "back"],
    pipeline: PipelineDep,
    file: Annotated[UploadFile, File()],
):
    logger.info("Enter to extract endpoint")
    image = image_to_ndarray(file)
    return pipeline.extract_side(image, side=side)
