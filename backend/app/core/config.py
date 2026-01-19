# @title 🛠️ Fix Configuration (Overwrite config.py)
import os

# We write the full file content with defaults and JSON loading logic
config_content = """
\"\"\"
Core Configuration Module (Patched for Dev).
Implements the 12-Factor App 'Config' principle.
Loads and validates environment variables using Pydantic Settings.
\"\"\"

from typing import List, Union, Optional, Dict, Any
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import os

class Settings(BaseSettings):
    \"\"\"
    Global Application Settings.
    Validates environment variables on startup.
    \"\"\"
    
    # --- 1. General Info ---
    PROJECT_NAME: str = "DECODE Energy Router"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # --- 2. Security & Auth ---
    # FIXED: Added default value for dev environment
    SECRET_KEY: str = "dev_secret_key_insecure_do_not_use_in_prod"
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

    # --- 3. Infrastructure (DB & Cache) ---
    # FIXED: Changed types to 'str' and added SQLite/Localhost defaults
    DATABASE_URL: str = "sqlite:///./dev.db" 
    REDIS_URL: str = "redis://localhost:6379"

    # --- 4. Physics & AI Engine ---
    MODEL_PATH: str = "assets/models/pinn_traced.pt"
    
    # Physics Dictionary (Populated from grid_parameters.json)
    PHYSICS: Dict[str, Any] = {}

    # --- 5. Edge Specifics ---
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
        \"\"\"
        Automatically loads grid_parameters.json to populate physics constraints.
        Searches common paths to ensure it works in Colab/CLI.
        \"\"\"
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
                        
                    # Flatten specific limits for easier access in main.py
                    limits = data.get("operational_limits", {}).get("voltage", {})
                    freq = data.get("operational_limits", {}).get("frequency", {})
                    
                    self.PHYSICS = {
                        "min_voltage_pu": limits.get("min_pu", 0.9),
                        "max_voltage_pu": limits.get("max_pu", 1.1),
                        "max_freq_dev_hz": freq.get("deadband_hz", 0.5),
                        "raw_data": data # Keep full data just in case
                    }
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed parsing physics config: {e}")

        if not loaded:
            print("⚠️ Physics Config Not Found. Using Hardcoded Safe Defaults.")
            self.PHYSICS = {"min_voltage_pu": 0.9, "max_voltage_pu": 1.1, "max_freq_dev_hz": 0.5}

# --- Singleton Instance ---
settings = Settings()
"""

# Overwrite the file
os.makedirs("backend/app/core", exist_ok=True)
with open("backend/app/core/config.py", "w") as f:
    f.write(config_content)

print("✅ backend/app/core/config.py patched successfully!")
