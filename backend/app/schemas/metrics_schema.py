"""
Metrics Schema Definitions.
Defines all observable outputs from the Digital Twin simulation.
Standardizes telemetry, power flow results, and reliability metrics.
Compatible with control_schema.py and model_output_schema.py.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime

# --- Enums for Metric Status ---
class SystemStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    FAULT = "fault"
    ISLANDED = "islanded"

class LoadStatus(str, Enum):
    SERVED = "served"
    SHED = "shed"
    CURTAILED = "curtailed"

# --- Power Flow Metrics ---
class BusMetrics(BaseModel):
    """
    Per-bus electrical measurements.
    Captures voltage, angle, and load status.
    """
    bus_id: int = Field(
        ge=0,
        description="Bus index in network topology."
    )
    
    voltage_pu: float = Field(
        ge=0.0,
        le=2.0,
        description="Voltage magnitude in per-unit (nominal = 1.0)."
    )
    
    voltage_angle_deg: float = Field(
        ge=-180.0,
        le=180.0,
        description="Voltage phase angle in degrees."
    )
    
    active_power_mw: float = Field(
        description="MW - Net active power injection (positive = generation)."
    )
    
    reactive_power_mvar: float = Field(
        description="MVAR - Net reactive power injection."
    )
    
    load_status: LoadStatus = Field(
        default=LoadStatus.SERVED,
        description="Load service status at this bus."
    )
    
    voltage_violation: bool = Field(
        default=False,
        description="True if voltage outside ANSI C84.1 limits (0.95-1.05 p.u.)."
    )

class LineMetrics(BaseModel):
    """
    Per-line power flow and thermal loading.
    """
    line_id: str = Field(
        description="Line identifier (e.g., 'Line_1-2')."
    )
    
    from_bus: int = Field(ge=0)
    to_bus: int = Field(ge=0)
    
    active_flow_mw: float = Field(
        description="MW - Active power flow (from_bus -> to_bus)."
    )
    
    reactive_flow_mvar: float = Field(
        description="MVAR - Reactive power flow."
    )
    
    current_amps: float = Field(
        ge=0.0,
        description="Current magnitude in amperes."
    )
    
    loading_percent: float = Field(
        ge=0.0,
        le=200.0,
        description="% of thermal rating (>100% = overload)."
    )
    
    is_overloaded: bool = Field(
        default=False,
        description="True if loading exceeds emergency rating."
    )

# --- Generation Metrics ---
class GeneratorMetrics(BaseModel):
    """
    Per-generator dispatch and headroom.
    """
    gen_id: str = Field(
        description="Generator identifier."
    )
    
    fuel_type: str = Field(
        default="unknown",
        description="Fuel source (solar, wind, gas, battery, etc.)."
    )
    
    active_output_mw: float = Field(
        ge=0.0,
        description="MW - Current active power output."
    )
    
    reactive_output_mvar: float = Field(
        description="MVAR - Current reactive power output."
    )
    
    capacity_mw: float = Field(
        ge=0.0,
        description="MW - Nameplate capacity."
    )
    
    availability_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="% of capacity currently available."
    )
    
    curtailment_mw: float = Field(
        default=0.0,
        ge=0.0,
        description="MW - Renewable energy curtailed due to grid constraints."
    )

# --- Battery Metrics ---
class BatteryMetrics(BaseModel):
    """
    Energy storage state and power flow.
    """
    battery_id: str = Field(
        description="Battery system identifier."
    )
    
    soc_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="% - State of Charge."
    )
    
    power_kw: float = Field(
        description="kW - Instantaneous power (positive = discharge, negative = charge)."
    )
    
    energy_kwh: float = Field(
        ge=0.0,
        description="kWh - Stored energy."
    )
    
    capacity_kwh: float = Field(
        ge=0.0,
        description="kWh - Total usable capacity."
    )
    
    charge_rate_kw: float = Field(
        ge=0.0,
        description="kW - Maximum charge rate."
    )
    
    discharge_rate_kw: float = Field(
        ge=0.0,
        description="kW - Maximum discharge rate."
    )
    
    cycles_count: int = Field(
        default=0,
        ge=0,
        description="Cumulative charge/discharge cycles."
    )
    
    health_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="% - Battery health/degradation metric."
    )

# --- Reliability Metrics ---
class ReliabilityMetrics(BaseModel):
    """
    System-level reliability and service quality indicators.
    """
    saidi_minutes: float = Field(
        default=0.0,
        ge=0.0,
        description="System Average Interruption Duration Index (minutes/year)."
    )
    
    saifi_count: float = Field(
        default=0.0,
        ge=0.0,
        description="System Average Interruption Frequency Index (interruptions/year)."
    )
    
    total_load_served_mwh: float = Field(
        ge=0.0,
        description="MWh - Energy delivered to customers."
    )
    
    total_load_shed_mwh: float = Field(
        default=0.0,
        ge=0.0,
        description="MWh - Unserved energy due to outages."
    )
    
    loss_of_load_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of failing to meet demand (0-1)."
    )
    
    reserve_margin_percent: float = Field(
        ge=0.0,
        description="% - Available generation headroom above peak demand."
    )

# --- Economic Metrics ---
class EconomicMetrics(BaseModel):
    """
    Cost and revenue tracking.
    """
    energy_cost_usd: float = Field(
        default=0.0,
        description="$ - Total energy procurement cost."
    )
    
    demand_charge_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="$ - Demand charge based on peak kW."
    )
    
    battery_degradation_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="$ - Estimated battery wear cost."
    )
    
    export_revenue_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="$ - Revenue from grid export."
    )
    
    total_cost_usd: float = Field(
        description="$ - Net operational cost (cost - revenue)."
    )
    
    carbon_emissions_kg: float = Field(
        default=0.0,
        ge=0.0,
        description="kg CO₂ - Carbon footprint of energy consumption."
    )

# --- Weather Metrics ---
class WeatherMetrics(BaseModel):
    """
    Environmental conditions affecting generation.
    """
    solar_irradiance_wm2: float = Field(
        ge=0.0,
        le=1500.0,
        description="W/m² - Global Horizontal Irradiance."
    )
    
    cloud_cover_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="% - Cloud cover fraction."
    )
    
    wind_speed_mps: float = Field(
        ge=0.0,
        description="m/s - Wind speed at hub height."
    )
    
    temperature_celsius: float = Field(
        description="°C - Ambient temperature."
    )
    
    forecast_accuracy_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="% - Accuracy of weather forecast vs. actual."
    )

# --- Master Metrics Container ---
class SystemMetrics(BaseModel):
    """
    Complete observable state of the Digital Twin.
    Aggregates all subsystem metrics into a single snapshot.
    """
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of measurement."
    )
    
    system_status: SystemStatus = Field(
        default=SystemStatus.NORMAL,
        description="Overall system health indicator."
    )
    
    # Component-level metrics (lists for scalability)
    buses: List[BusMetrics] = Field(
        default_factory=list,
        description="Per-bus voltage and power metrics."
    )
    
    lines: List[LineMetrics] = Field(
        default_factory=list,
        description="Per-line flow and thermal loading."
    )
    
    generators: List[GeneratorMetrics] = Field(
        default_factory=list,
        description="Per-generator dispatch and availability."
    )
    
    batteries: List[BatteryMetrics] = Field(
        default_factory=list,
        description="Energy storage state metrics."
    )
    
    # System-level aggregates
    reliability: ReliabilityMetrics = Field(
        default_factory=ReliabilityMetrics,
        description="Reliability indices (SAIDI/SAIFI)."
    )
    
    economics: EconomicMetrics = Field(
        default_factory=EconomicMetrics,
        description="Cost and carbon metrics."
    )
    
    weather: WeatherMetrics = Field(
        default_factory=WeatherMetrics,
        description="Environmental conditions."
    )
    
    # Simulation metadata
    simulation_id: Optional[str] = Field(
        default=None,
        description="Unique identifier linking to scenario."
    )
    
    convergence_error: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Power flow solver residual (None = converged)."
    )

    # --- Derived Properties ---
    @property
    def total_generation_mw(self) -> float:
        """Sum of all active generator outputs."""
        return sum(gen.active_output_mw for gen in self.generators)
    
    @property
    def total_load_mw(self) -> float:
        """Sum of all bus load injections."""
        return sum(abs(bus.active_power_mw) for bus in self.buses if bus.active_power_mw < 0)
    
    @property
    def average_voltage_pu(self) -> float:
        """Mean voltage across all buses."""
        if not self.buses:
            return 1.0
        return sum(bus.voltage_pu for bus in self.buses) / len(self.buses)
    
    @property
    def has_violations(self) -> bool:
        """True if any constraint violations exist."""
        return (
            any(bus.voltage_violation for bus in self.buses) or
            any(line.is_overloaded for line in self.lines) or
            self.system_status in [SystemStatus.WARNING, SystemStatus.CRITICAL, SystemStatus.FAULT]
        )

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-01-20T14:30:00Z",
                "system_status": "normal",
                "buses": [
                    {
                        "bus_id": 1,
                        "voltage_pu": 1.02,
                        "voltage_angle_deg": 0.0,
                        "active_power_mw": 5.2,
                        "reactive_power_mvar": 1.1,
                        "load_status": "served",
                        "voltage_violation": False
                    }
                ],
                "reliability": {
                    "saidi_minutes": 45.2,
                    "saifi_count": 1.2,
                    "reserve_margin_percent": 15.0
                },
                "economics": {
                    "total_cost_usd": 1250.50,
                    "carbon_emissions_kg": 320.0
                }
            }
  }
