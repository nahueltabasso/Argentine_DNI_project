from pydantic import BaseModel, Field
from datetime import datetime

class FrontIDData(BaseModel):
    
    doc_number: str | None = Field(None, description="Document number")
    tramite_number: str | None = Field(None, description="Tramite number")
    has_shield: bool = Field(..., description="Shield status")
    pdf417: dict | None = Field(None, description="PDF 417 data")
    mrz: dict | None = Field(None, description="MRZ data")
    gender: str | None = Field(None, description="Gender")
    
class BackIDData(BaseModel):
    
    address: str | None = Field(None, description="Address")
    has_country: bool | None = Field(None, description="Country")
    mrz: dict | None = Field(None, description="MRZ data")
    pdf417: dict | None = Field(None, description="PDF 417 data")
    tramite_number: str | None = Field(None, description="Tramite number")
    
class ResponseData(BaseModel):
    
    front_coords: list[list[int]] | None = Field(None, description="Front document coordinates")
    back_coords: list[list[int]] | None = Field(None, description="Back document coordinates")
    is_dni: bool | None = Field(None, description="Indicates if the document is a DNI")
    front_data: FrontIDData | None = Field(None, description="Front ID data")
    back_data: BackIDData | None = Field(None, description="Back ID data")
    confidence: float = Field(0.0, description="Confidence level", ge=0.0, le=1.0)
    timestamp: datetime = Field(..., description="Timestamp of the response")