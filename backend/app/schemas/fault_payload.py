"""
Fault Scenario Schemas.
Defines the structure and validation rules for simulation requests.
Ensures only physically plausible scenarios are passed to the Digital Twin.
Unifies legacy access (main.py) with modern validation (simulation.py).
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from datetime import datetime

# --- Enums for Strict Typing ---
class FaultType(str, Enum):
    THREE_PHASE = "3ph_short"
    LINE_TO_GROUND = "1ph_ground"
    LOSS_OF_GENERATION = "gen_trip"
    LOAD_SPIKE = "load_surge"
    CLOUD_COVER = "cloud_gorilla" 
    NONE = "none"

class ComponentType(str, Enum):
    LINE = "line"
    BUS = "bus"
    TRANSFORMER = "transformer"
    GENERATOR = "generator"
    UNKNOWN = "unknown"

class FaultPayload(BaseModel):
    """
    Unified Payload for triggering a 'What-If' contingency analysis.
    Compatible with legacy 'main.py' physics checks and modern 'simulation.py' routes.
    """
    # --- Core Fields ---
    fault_type: FaultType = Field(
        default=FaultType.NONE, 
        description="The physics of the failure mode."
    )
    
    magnitude: float = Field(
        default=0.0, 
        ge=0.0, 
        le=10.0, 
        description="Severity of the fault (0.0 to 1.0 p.u. or multiplier)."
    )
    
    location_bus: int = Field(
        default=0, 
        ge=0, 
        description="Bus index where fault occurred."
    )
    
    duration_sec: float = Field(
        default=0.0, 
        ge=0.0, 
        description="Duration of the fault in seconds."
    )

    # --- Metadata (Optional) ---
    timestamp: Optional[datetime] = Field(
        default=None, 
        description="UTC timestamp for the simulation start."
    )
    
    target_component: Optional[str] = Field(
        default=None, 
        example="Line_1-2", 
        description="ID of the grid asset to fail."
    )

    # --- COMPATIBILITY PROPERTIES (The Fix) ---
    # These properties allow main.py to access attributes it expects 
    # without needing to rewrite the main logic.

    @property
    def is_active(self) -> bool:
        """
        Legacy compatibility for main.py: 
        Checks if fault is actually active based on magnitude and type.
        """
        return self.magnitude > 0 and self.fault_type != FaultType.NONE

    @property
    def magnitude_pu(self) -> float:
        """
        Legacy compatibility for main.py: 
        Alias for 'magnitude' to satisfy legacy physics calculations.
        """
        return self.magnitude

    # --- Validators ---
    @field_validator('target_component')
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        if v and (len(v) > 50 or " " in v):
            raise ValueError("Invalid Component ID format.")
        return v
