"""
Core Configuration Module.
Implements the 12-Factor App 'Config' principle.
Loads and validates environment variables using Pydantic Settings.
"""

from typing import List, Union, Optional, Dict, Any
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    """
    Global Application Settings.
    Validates environment variables on startup.
    """
    
    # --- 1. General Info ---
    PROJECT_NAME: str = "DECODE Energy Router"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment: development, staging, production
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # --- 2. Security & Auth ---
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS (Cross-Origin Resource Sharing)
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parses a comma-separated string of origins into a list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # --- 3. Database (PostgreSQL) - Optional for MVP ---
    DATABASE_URL: Optional[str] = None

    # --- 4. Cache & Queue (Redis) - Optional for MVP ---
    REDIS_URL: Optional[str] = None

    # --- 5. Physics & AI Engine ---
    MODEL_PATH: str = "assets/models/pinn_traced.pt"
    
    # Physics Tolerances (Governance)
    MAX_VOLTAGE_PU: float = 1.05
    MIN_VOLTAGE_PU: float = 0.95
    PHYSICS_RESIDUAL_THRESHOLD: float = 1e-3
    
    # Physics Dictionary for backward compatibility
    PHYSICS: Dict[str, float] = {
        "min_voltage_pu": 0.9,
        "max_freq_dev_hz": 0.5
    }

    # --- 6. Edge Specifics ---
    EDGE_DEVICE_ID: str = "dev_local"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

# --- Singleton Instance ---
settings = Settings()

# --- Logging Configuration ---
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO" if settings.ENVIRONMENT == "production" else "DEBUG",
        "handlers": ["console"],
    },
}
