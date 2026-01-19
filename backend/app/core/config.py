
from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "D.E.C.O.D.E."
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "dev_secret"
    DATABASE_URL: Optional[str] = "sqlite:///./dev.db"
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"
    MODEL_PATH: str = "assets/models/pinn_traced.pt"
    
    PHYSICS: Dict[str, float] = {
        "min_voltage_pu": 0.9, "max_voltage_pu": 1.1,
        "max_freq_dev_hz": 0.5, "base_power_mva": 100.0, "nominal_freq_hz": 60.0
    }
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
