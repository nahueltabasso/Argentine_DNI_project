from fastapi import APIRouter, File, UploadFile, status
from app.dependencies.dependency import DocumentDetectorDep, ElementsServiceDep
from app.schemas.schemas import ArgentineIDData, DocumentDetected
from app.utils.file_utils import image_to_ndarray
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/document/detect", status_code=status.HTTP_200_OK, response_model=DocumentDetected)
def detect_document(
    detector: DocumentDetectorDep,
    file: UploadFile = File(...)
):
    logger.info("Enter to detect_document endpoint")
    image = image_to_ndarray(file)
    response = detector.get_argentine_ID_card(image)
    return response

@router.post("/document/extract-full",  status_code=status.HTTP_200_OK, response_model=ArgentineIDData)
def extract_full_document(
    service: ElementsServiceDep,
    file: UploadFile = File(...),
    file1: UploadFile = File(...)
):
    logger.info("Enter to extract_full_document endpoint")
    image = image_to_ndarray(file)
    image1 = image_to_ndarray(file1)
    response = service.get_data_from_doc(image, image1)
    return response

# @router.post("/document/extract", status_code=status.HTTP_200_OK, response_model=ArgentineIDData)
# def 