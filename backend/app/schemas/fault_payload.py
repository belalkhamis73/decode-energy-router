# @title 🛠️ Fix 1: Update Schemas (fault_payload.py)
import os

schema_content = """
\"\"\"
Fault Scenario Schemas.
Defines the structure and validation rules for simulation requests.
Ensures only physically plausible scenarios are passed to the Digital Twin.
\"\"\"

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum  # <--- CRITICAL IMPORT

# --- Enums for Strict Typing ---
# FIXED: Inherit from (str, Enum) so Pydantic v2 generates valid schemas
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
    CLOUD_COVER = "cloud_gorilla" 

# --- Input Schema (Request) ---
class FaultScenario(BaseModel):
    \"\"\"
    Payload for triggering a 'What-If' contingency analysis.
    \"\"\"
    timestamp: datetime = Field(..., description="UTC timestamp for the simulation start.")
    target_component: str = Field(..., example="Line_1-2", description="ID of the grid asset to fail.")
    
    # Use the Enum class as the type annotation
    component_type: ComponentType = Field(..., description="Category of the asset.")
    
    fault_type: FaultType = Field(..., description="The physics of the failure mode.")
    
    magnitude: float = Field(
        1.0, 
        ge=0.0, 
        le=1.0, 
        description="Severity of the fault (0.0 to 1.0)."
    )
    
    duration_ms: int = Field(
        100, 
        ge=1, 
        le=5000, 
        description="Duration of the fault in milliseconds."
    )

    @field_validator('target_component')
    @classmethod
    def validate_component_id(cls, v: str) -> str:
        if len(v) > 50 or " " in v:
            raise ValueError("Invalid Component ID format.")
        return v

# --- Output Schema (Response) ---
class GridState(BaseModel):
    voltages: List[float]
    frequency: float
    stability_margin: float

class SimulationResult(BaseModel):
    timestamp: datetime
    grid_state: GridState
    physics_violation_detected: bool
    computation_time_ms: Optional[float] = None
"""

os.makedirs("backend/app/schemas", exist_ok=True)
with open("backend/app/schemas/fault_payload.py", "w") as f:
    f.write(schema_content)
print("✅ Fixed fault_payload.py")
