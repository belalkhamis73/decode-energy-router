"""
Core Configuration Module.
Implements the 12-Factor App 'Config' principle.
Loads and validates environment variables using Pydantic Settings.
"""

from typing import List, Union, Optional, Dict, Any
from pydantic import AnyHttpUrl, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets
import json
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
    # Defaults to 'production' for safety
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    # --- 2. Security & Auth ---
    # Secret key for JWT signature. Generate a strong one in prod!
    # Fail Fast: Raises error if missing in production
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS (Cross-Origin Resource Sharing)
    # List of origins allowed to call the API (e.g., frontend URL)
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parses a comma-separated string of origins into a list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # --- 3. Database (PostgreSQL) ---
    # Connection string for the Grid Topology Store
    DATABASE_URL: PostgresDsn

    # --- 4. Cache & Queue (Redis) ---
    # Connection string for State Management & Celery Broker
    REDIS_URL: RedisDsn

    # --- 5. Physics & AI Engine ---
    # Path to the trained PyTorch model artifact
    # If strictly running on edge, this might be a local path
    MODEL_PATH: str = "assets/models/pinn_traced.pt"
    
    # Physics Tolerances (Governance)
    MAX_VOLTAGE_PU: float = 1.05
    MIN_VOLTAGE_PU: float = 0.95
    PHYSICS_RESIDUAL_THRESHOLD: float = 1e-3

    # --- 6. Edge Specifics ---
    # Device ID for distributed fleet management
    EDGE_DEVICE_ID: str = "dev_local"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True,
        # In production, extra fields are ignored or warned
        extra="ignore" 
    )

# --- Singleton Instance ---
# Use lru_cache if reading from file often, but simple instantiation works for env vars
settings = Settings()

# --- Logging Configuration ---
# (Simple setup for 12-factor stdout logging)
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
  
