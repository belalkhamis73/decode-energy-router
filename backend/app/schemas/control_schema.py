"""
Control Schema Definitions.
Defines the user-facing control surface for the Digital Twin.
Allows operators to override weather, control batteries, manage grid, and set optimization goals.
Compatible with fault_payload.py and the broader simulation framework.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from enum import Enum
from datetime import datetime

# --- Enums for Control Modes ---
class BatteryMode(str, Enum):
    AUTO = "auto"
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"

class GridMode(str, Enum):
    ISLANDED = "islanded"
    GRID_TIED = "grid_tied"
    EXPORT_PRIORITY = "export_priority"
    IMPORT_PRIORITY = "import_priority"

class OptimizationObjective(str, Enum):
    COST_MIN = "minimize_cost"
    RELIABILITY_MAX = "maximize_reliability"
    CARBON_MIN = "minimize_carbon"
    BALANCED = "balanced"

# --- Weather Override Controls ---
class WeatherOverrides(BaseModel):
    """
    Allow operators to inject synthetic weather conditions.
    Useful for stress-testing solar/wind forecasts.
    """
    solar_irradiance_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1500.0,
        description="W/m² - Override GHI (Global Horizontal Irradiance). None = use forecast."
    )
    
    cloud_cover_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="% cloud cover for solar degradation testing."
    )
    
    wind_speed_mps: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50.0,
        description="m/s - Override wind speed. None = use forecast."
    )
    
    temperature_celsius: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=60.0,
        description="°C - Ambient temperature for thermal derating."
    )
    
    apply_from: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when override begins."
    )
    
    duration_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=168.0,
        description="Hours to maintain override (max 1 week)."
    )

# --- Battery Energy Storage Controls ---
class BatteryControls(BaseModel):
    """
    Direct control over battery dispatch and SOC targets.
    Overrides optimization when manual mode is engaged.
    """
    mode: BatteryMode = Field(
        default=BatteryMode.AUTO,
        description="Operating mode: auto (optimizer), manual charge/discharge, or idle."
    )
    
    power_setpoint_kw: float = Field(
        default=0.0,
        ge=-10000.0,
        le=10000.0,
        description="kW - Positive = discharge, Negative = charge. Clamped to battery limits."
    )
    
    soc_target_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="% - Target State of Charge. Used in AUTO mode for schedule optimization."
    )
    
    reserve_margin_percent: float = Field(
        default=20.0,
        ge=0.0,
        le=50.0,
        description="% - Minimum SOC reserve for grid services/backup."
    )
    
    enable_grid_services: bool = Field(
        default=True,
        description="Allow battery to participate in frequency regulation."
    )

    @field_validator('power_setpoint_kw')
    @classmethod
    def validate_power_sign(cls, v: float, info) -> float:
        """Ensure power setpoint respects mode."""
        mode = info.data.get('mode')
        if mode == BatteryMode.CHARGE and v > 0:
            raise ValueError("CHARGE mode requires negative power_setpoint_kw")
        if mode == BatteryMode.DISCHARGE and v < 0:
            raise ValueError("DISCHARGE mode requires positive power_setpoint_kw")
        return v

# --- Grid Interconnection Controls ---
class GridControls(BaseModel):
    """
    Manage grid connection, import/export limits, and islanding behavior.
    """
    mode: GridMode = Field(
        default=GridMode.GRID_TIED,
        description="Grid interaction mode."
    )
    
    import_limit_kw: float = Field(
        default=5000.0,
        ge=0.0,
        le=50000.0,
        description="kW - Maximum power draw from utility grid."
    )
    
    export_limit_kw: float = Field(
        default=5000.0,
        ge=0.0,
        le=50000.0,
        description="kW - Maximum power export to utility grid."
    )
    
    enable_blackstart: bool = Field(
        default=False,
        description="Allow microgrid to restart without utility grid."
    )
    
    voltage_setpoint_pu: float = Field(
        default=1.0,
        ge=0.9,
        le=1.1,
        description="p.u. - Target voltage at PCC (Point of Common Coupling)."
    )
    
    frequency_deadband_hz: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Hz - Frequency regulation deadband."
    )

# --- Optimization Weights ---
class OptimizationWeights(BaseModel):
    """
    Multi-objective optimization tuning knobs.
    Defines tradeoffs between cost, reliability, and carbon.
    """
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.BALANCED,
        description="Primary optimization goal."
    )
    
    cost_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for operational cost minimization (0-1)."
    )
    
    reliability_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for reliability/SAIDI minimization (0-1)."
    )
    
    carbon_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for carbon emissions minimization (0-1)."
    )
    
    time_horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Optimization lookahead window (1-168 hours)."
    )

    @field_validator('carbon_weight')
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """Ensure weights sum to ~1.0 for normalized objective."""
        cost = info.data.get('cost_weight', 0)
        reliability = info.data.get('reliability_weight', 0)
        total = cost + reliability + v
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Weights must sum to ~1.0, got {total:.2f}")
        return v

# --- Master Control Surface ---
class UserControlSurface(BaseModel):
    """
    Unified control interface for Digital Twin operators.
    Combines weather, battery, grid, fault, and optimization controls.
    """
    weather_overrides: WeatherOverrides = Field(
        default_factory=WeatherOverrides,
        description="Synthetic weather injection."
    )
    
    battery_controls: BatteryControls = Field(
        default_factory=BatteryControls,
        description="Battery dispatch and SOC management."
    )
    
    grid_controls: GridControls = Field(
        default_factory=GridControls,
        description="Grid interconnection settings."
    )
    
    optimization_weights: OptimizationWeights = Field(
        default_factory=OptimizationWeights,
        description="Objective function tuning."
    )
    
    simulation_name: str = Field(
        default="Untitled Scenario",
        max_length=100,
        description="Human-readable scenario identifier."
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when control surface was created."
    )
    
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Operator notes/justification for control changes."
    )

    # --- Legacy Compatibility ---
    @property
    def has_overrides(self) -> bool:
        """Check if any manual overrides are active."""
        return (
            any([
                self.weather_overrides.solar_irradiance_override is not None,
                self.weather_overrides.wind_speed_mps is not None,
                self.battery_controls.mode != BatteryMode.AUTO,
                self.grid_controls.mode != GridMode.GRID_TIED
            ])
        )

    class Config:
        json_schema_extra = {
            "example": {
                "simulation_name": "Hurricane Stress Test",
                "weather_overrides": {
                    "wind_speed_mps": 35.0,
                    "cloud_cover_percent": 100.0,
                    "duration_hours": 6.0
                },
                "battery_controls": {
                    "mode": "discharge",
                    "soc_target_percent": 30.0
                },
                "grid_controls": {
                    "mode": "islanded"
                },
                "optimization_weights": {
                    "objective": "maximize_reliability",
                    "reliability_weight": 0.7,
                    "cost_weight": 0.2,
                    "carbon_weight": 0.1
                }
            }
  }
