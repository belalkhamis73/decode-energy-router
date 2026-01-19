# @title 🛠️ Fix Configuration (Overwrite config.py)
import os

config_content = """
\"\"\"
Core Configuration Module (Patched for Dev).
\"\"\"

from typing import List, Union, Optional, Dict, Any
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import os

class Settings(BaseSettings):
    # --- 1. General Info (FIXED: Added PROJECT_NAME) ---
    PROJECT_NAME: str = "DECODE Energy Router"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # --- 2. Security & Auth ---
    SECRET_KEY: str = "dev_insecure_secret_key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # --- 3. Infrastructure ---
    DATABASE_URL: str = "sqlite:///./dev.db" 
    REDIS_URL: str = "redis://localhost:6379"

    # --- 4. Physics Engine ---
    MODEL_PATH: str = "assets/models/pinn_traced.pt"
    PHYSICS: Dict[str, Any] = {}
    EDGE_DEVICE_ID: str = "dev_local"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore" 
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_physics_constants()

    def _load_physics_constants(self):
        search_paths = [
            "physics_core/constants/grid_parameters.json",
            "../physics_core/constants/grid_parameters.json",
            "/content/decode-energy-router/physics_core/constants/grid_parameters.json"
        ]
        
        loaded = False
        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    limits = data.get("operational_limits", {}).get("voltage", {})
                    self.PHYSICS = {
                        "min_voltage_pu": limits.get("min_pu", 0.9),
                        "max_voltage_pu": limits.get("max_pu", 1.1),
                        "max_freq_dev_hz": 0.5,
                        "raw_data": data 
                    }
                    loaded = True
                    break
                except Exception:
                    pass

        if not loaded:
            self.PHYSICS = {"min_voltage_pu": 0.9, "max_voltage_pu": 1.1, "max_freq_dev_hz": 0.5}

settings = Settings()
"""

os.makedirs("backend/app/core", exist_ok=True)
with open("backend/app/core/config.py", "w") as f:
    f.write(config_content)
print("✅ Fixed config.py (Restored PROJECT_NAME)")
