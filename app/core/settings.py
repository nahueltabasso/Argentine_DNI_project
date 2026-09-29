from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # --- APP ---
    app_env: Literal["local", "docker", "prod"] = "docker"
    api_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    load_models: bool = True
    
    # --- Argentine ID-Card Detector (YOLO OBB) ---
    yolo_dni_detector: Path
    doc_conf: float = 0.6
    doc_iou: float = 0.5
    doc_imgsz: int = 640
    
    # --- Elements Detector (YOLO Detect) ---
    yolo_id_elements_detector: Path
    elem_conf: float = 0.6
    elem_iou: float = 0.5
    elem_imgsz: int = 1216
    
    # --- Inference ---
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    
    # --- Address / VLM ---
    address_strategy: Literal["paddleocr", "vlm"] = "vlm"
    google_api_key: SecretStr | None = None
    vlm_model: str = "gemini-3.6-flash"
    vlm_timeout_s: float = 30.0
    
    @model_validator(mode="after") # type: ignore
    def check_vlm_key(self) -> "Settings": 
        if self.address_strategy == "vlm" and self.google_api_key is None:
            raise ValueError("Google API key must be provided when using VLM address strategy.")
        return self
    
    
@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore