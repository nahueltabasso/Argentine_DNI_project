from enum import IntEnum, StrEnum
from pydantic import BaseModel, ConfigDict, Field, computed_field
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

    side: str = Field("", description="Document side, e.g., 'front' or 'back'")
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

    @computed_field
    @property
    def match_confidence(self) -> float: 
        """Calculate match confidence between document fields and extracted data"""
        pdf417 = self.pdf417 or {}
        mrz_number = self.mrz.document_number if self.mrz else None
        doc_number = self.doc_number.replace(".", "").replace(",", "") if self.doc_number else None
        
        checks = [
            (doc_number, mrz_number),
            (doc_number, pdf417.get("document_number")),
            (pdf417.get("document_number"), mrz_number),
            (self.tramite_number, pdf417.get("tramite_number")),
        ]

        comparable = [(a, b) for a, b in checks if a and b]
        if not comparable:
            return 0.0

        return sum(a == b for a, b in comparable) / len(comparable)

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
    