"""
Fault Scenario Schemas.
Defines the structure and validation rules for simulation requests.
Ensures only physically plausible scenarios are passed to the Digital Twin.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum  # <--- FIXED: Added Enum Import

# --- Enums for Strict Typing ---
# FIXED: Inherit from (str, Enum) so Pydantic recognizes them as valid types
class ComponentType(str, Enum):
    LINE = "line"
    BUS = "bus"
    TRANSFORMER = "transformer"
    GENERATOR = "generator"

class FaultType(str, Enum):
    THREE_PHASE = "3ph_short"
    LINE_TO_GROUND = "1ph_ground"
    LOSS_OF_GENERATION = "gen_trip"
    LOAD_SPIKE = "load_surge"
    CLOUD_COVER = "cloud_gorilla" # 80% PV drop in <10s

# --- Input Schema (Request) ---
class FaultScenario(BaseModel):
    """
    Payload for triggering a 'What-If' contingency analysis.
    """
    timestamp: datetime = Field(..., description="UTC timestamp for the simulation start.")
    target_component: str = Field(..., example="Line_1-2", description="ID of the grid asset to fail.")
    
    # FIXED: Use the Enum class here instead of Literal for better validation
    component_type: ComponentType = Field(..., description="Category of the asset.")
    
    fault_type: FaultType = Field(..., description="The physics of the failure mode.")
    
    magnitude: float = Field(
        1.0, 
        ge=0.0, 
        le=1.0, 
        description="Severity of the fault (0.0 to 1.0). E.g., 0.8 means 80% voltage dip or cloud cover."
    )
    
    duration_ms: int = Field(
        100, 
        ge=1, 
        le=5000, 
        description="Duration of the fault in milliseconds. Capped at 5s for stability."
    )

    @field_validator('target_component')
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        """Sanity check to prevent injection or malformed IDs."""
        if len(v) > 50 or " " in v:
            raise ValueError("Invalid Component ID format.")
        return v

# --- Output Schema (Response) ---
class GridState(BaseModel):
    """
    Snapshot of the grid's physical health.
    """
    voltages: List[float] = Field(..., description="Per-unit voltage magnitude at each bus.")
    frequency: float = Field(..., description="System frequency in Hz.")
    stability_margin: float = Field(..., description="Distance to voltage collapse (Voltage Stability Index).")

class SimulationResult(BaseModel):
    """
    The Digital Twin's verdict on the scenario.
    """
    timestamp: datetime
    grid_state: GridState
    physics_violation_detected: bool = Field(..., description="True if the scenario caused a crash (e.g., Voltage < 0.7 p.u).")
    
    # Optional metadata for debugging
    computation_time_ms: Optional[float] = None
