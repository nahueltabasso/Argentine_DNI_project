from enum import IntEnum, StrEnum
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from dataclasses import dataclass
import numpy as np

class MRZData(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    surname: str | None = None
    name: str | None = None
    country: str | None = None
    nationality: str | None = None
    birth_date: str | None = None
    expiry_date: str | None = None
    sex: str | None = None
    document_type: str | None = None
    document_number: str | None = None
    optional_data: str | None = None
    is_valid: bool = False

class ArgentineIDData(BaseModel):

    doc_number: str | None = Field(None, description="Document number")
    tramite_number: str | None = Field(None, description="Tramite number")
    has_shield: bool | None = Field(None, description="Shield status")
    has_picture: bool | None = Field(None, description="Picture status")
    pdf417: dict | None = Field(None, description="PDF 417 data")
    mrz: MRZData | None = Field(None, description="MRZ data")
    gender: str | None = Field(None, description="Gender")
    address: str | None = Field(None, description="Address")
    has_country: bool | None = Field(None, description="Country")
    timestamp: datetime = Field(..., description="Timestamp of the document")


class FrontIDData(BaseModel):
    
    doc_number: str | None = Field(None, description="Document number")
    tramite_number: str | None = Field(None, description="Tramite number")
    has_shield: bool | None = Field(False, description="Shield status")
    pdf417: dict | None = Field(None, description="PDF 417 data")
    mrz: dict | None = Field(None, description="MRZ data")
    gender: str | None = Field(None, description="Gender")
    timestamp: datetime = Field(..., description="Timestamp of the document side")

class BackIDData(BaseModel):
    
    address: str | None = Field(None, description="Address")
    has_country: bool | None = Field(None, description="Country")
    mrz: dict | None = Field(None, description="MRZ data")
    pdf417: dict | None = Field(None, description="PDF 417 data")
    tramite_number: str | None = Field(None, description="Tramite number")
    
class DocumentDetected(BaseModel):
    side: str = Field(..., description="Document side, e.g., 'front' or 'back'")
    points: list[list[int]] = Field(..., description="Coordinates of the document side")
    confidence: float = Field(..., description="Confidence level of the document side")
    timestamp: datetime = Field(..., description="Timestamp of the document side")
    
    
@dataclass  
class ElementDetection:
    
    box: list[int] 
    conf: float 
    crop: np.ndarray 

class SidesName(StrEnum):
    
    FRONT = "front"
    BACK = "back"

class ElementsID(IntEnum):
    
    PICTURE_CLS = 0
    SHIELD_CLS = 1
    DOC_NUMBER_CLS = 2
    TRAMITE_NUMBER_CLS = 3
    PDF417_CLS = 4
    MRZ_CLS = 5
    ADDRESS_CLS = 6
    GENDER_CLS = 7
    COUNTRY_CLS = 8
    