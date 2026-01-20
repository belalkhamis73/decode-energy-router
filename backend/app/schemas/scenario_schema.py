"""
Scenario Schema Definitions.
Defines comprehensive scenario injection payloads for Digital Twin testing.
Extends fault_payload.py with multi-fault sequences and complex contingencies.
Compatible with control_schema.py and metrics_schema.py.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime, timedelta
import sys
sys.path.append('.')
from fault_payload import FaultType, FaultPayload

# --- Extended Fault Types ---
class ScenarioType(str, Enum):
    SINGLE_FAULT = "single_fault"
    CASCADING = "cascading_failure"
    N_MINUS_K = "n_minus_k_contingency"
    EXTREME_WEATHER = "extreme_weather"
    CYBER_ATTACK = "cyber_attack"
    MARKET_STRESS = "market_stress"
    MAINTENANCE_OUTAGE = "planned_outage"
    COMPOSITE = "composite_scenario"

class WeatherSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"

class AttackVector(str, Enum):
    SCADA_INJECTION = "scada_false_data"
    COMMUNICATION_LOSS = "comms_denial"
    METERING_CORRUPTION = "meter_tampering"
    NONE = "none"

# --- Fault Event (Timestamped) ---
class FaultEvent(BaseModel):
    """
    Single fault occurrence with timing information.
    Used to build cascading or sequential fault scenarios.
    """
    fault: FaultPayload = Field(
        description="The fault instance to inject."
    )
    
    trigger_time_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds from scenario start when fault triggers."
    )
    
    clear_time_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Seconds from scenario start when fault clears (None = permanent)."
    )
    
    event_id: str = Field(
        default="event_0",
        max_length=50,
        description="Unique event identifier within scenario."
    )

    @field_validator('clear_time_sec')
    @classmethod
    def validate_clear_after_trigger(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure fault clears after it triggers."""
        if v is not None:
            trigger = info.data.get('trigger_time_sec', 0.0)
            if v <= trigger:
                raise ValueError(f"clear_time ({v}) must be > trigger_time ({trigger})")
        return v

# --- Weather Scenario ---
class WeatherScenario(BaseModel):
    """
    Extreme weather event with temporal evolution.
    Can be combined with faults for realistic storm scenarios.
    """
    severity: WeatherSeverity = Field(
        default=WeatherSeverity.MODERATE,
        description="Storm intensity classification."
    )
    
    wind_peak_mps: float = Field(
        default=15.0,
        ge=0.0,
        le=75.0,
        description="m/s - Peak wind speed during event."
    )
    
    precipitation_mm_per_hour: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="mm/hr - Rainfall intensity (affects line failures)."
    )
    
    lightning_strikes_per_hour: int = Field(
        default=0,
        ge=0,
        description="Lightning events (increases fault probability)."
    )
    
    solar_availability_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="% - Solar output degradation due to cloud cover."
    )
    
    start_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when weather event begins."
    )
    
    duration_hours: float = Field(
        default=2.0,
        ge=0.0,
        le=72.0,
        description="Hours - Event duration."
    )

# --- Cyber Attack Scenario ---
class CyberScenario(BaseModel):
    """
    Adversarial injection targeting SCADA/control systems.
    Simulates false data injection or communication disruption.
    """
    attack_vector: AttackVector = Field(
        default=AttackVector.NONE,
        description="Type of cyber threat."
    )
    
    target_components: List[str] = Field(
        default_factory=list,
        description="List of component IDs targeted by attack."
    )
    
    measurement_bias_percent: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0,
        description="% - Bias injected into measurements (positive = overreport)."
    )
    
    communication_loss_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability (0-1) of losing telemetry from target."
    )
    
    attack_start_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds from scenario start when attack begins."
    )
    
    attack_duration_sec: float = Field(
        default=60.0,
        ge=0.0,
        description="Seconds - Attack persistence."
    )

# --- N-k Contingency Scenario ---
class NMinusKScenario(BaseModel):
    """
    Simultaneous loss of K components (generalized N-1, N-2 analysis).
    Tests grid resilience to multiple concurrent failures.
    """
    k_count: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of components to fail simultaneously."
    )
    
    target_components: List[str] = Field(
        min_length=1,
        description="List of component IDs to fail (length must match k_count)."
    )
    
    failure_mode: FaultType = Field(
        default=FaultType.THREE_PHASE,
        description="Type of fault applied to all targets."
    )
    
    stagger_time_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds - Delay between successive failures (0 = simultaneous)."
    )

    @field_validator('target_components')
    @classmethod
    def validate_k_match(cls, v: List[str], info) -> List[str]:
        """Ensure number of targets matches k_count."""
        k = info.data.get('k_count', 1)
        if len(v) != k:
            raise ValueError(f"target_components length ({len(v)}) must equal k_count ({k})")
        return v

# --- Master Scenario Payload ---
class ScenarioPayload(BaseModel):
    """
    Comprehensive test scenario combining faults, weather, cyber, and contingencies.
    Extends FaultPayload with rich temporal and multi-domain scenarios.
    """
    scenario_id: str = Field(
        default="scenario_0",
        max_length=100,
        description="Unique scenario identifier."
    )
    
    scenario_type: ScenarioType = Field(
        default=ScenarioType.SINGLE_FAULT,
        description="Classification of scenario complexity."
    )
    
    description: str = Field(
        default="Untitled Scenario",
        max_length=500,
        description="Human-readable scenario description."
    )
    
    # --- Fault Sequence ---
    fault_events: List[FaultEvent] = Field(
        default_factory=list,
        description="Ordered list of fault injections (for cascading/sequential faults)."
    )
    
    # --- Optional Contextual Scenarios ---
    weather: Optional[WeatherScenario] = Field(
        default=None,
        description="Weather conditions concurrent with faults."
    )
    
    cyber: Optional[CyberScenario] = Field(
        default=None,
        description="Cyber attack overlay on physical faults."
    )
    
    n_minus_k: Optional[NMinusKScenario] = Field(
        default=None,
        description="Multi-component contingency analysis."
    )
    
    # --- Timing ---
    simulation_start: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp for scenario start."
    )
    
    total_duration_sec: float = Field(
        default=300.0,
        ge=0.0,
        le=86400.0,
        description="Seconds - Total simulation time (max 24 hours)."
    )
    
    time_step_sec: float = Field(
        default=1.0,
        ge=0.01,
        le=60.0,
        description="Seconds - Simulation time step resolution."
    )
    
    # --- Metadata ---
    created_by: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Operator who created scenario."
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Searchable tags (e.g., ['hurricane', 'islanding', 'battery_test'])."
    )

    # --- Legacy Compatibility with fault_payload.py ---
    @property
    def primary_fault(self) -> Optional[FaultPayload]:
        """Return first fault event for legacy single-fault API."""
        if self.fault_events:
            return self.fault_events[0].fault
        return None

    @property
    def is_complex_scenario(self) -> bool:
        """True if scenario involves multiple faults or non-fault disturbances."""
        return (
            len(self.fault_events) > 1 or
            self.weather is not None or
            self.cyber is not None or
            self.n_minus_k is not None
        )

    # --- Validators ---
    @field_validator('fault_events')
    @classmethod
    def validate_event_timing(cls, v: List[FaultEvent], info) -> List[FaultEvent]:
        """Ensure all fault events occur within simulation duration."""
        duration = info.data.get('total_duration_sec', float('inf'))
        for event in v:
            if event.trigger_time_sec > duration:
                raise ValueError(
                    f"Event '{event.event_id}' trigger ({event.trigger_time_sec}s) "
                    f"exceeds simulation duration ({duration}s)"
                )
        return v

    @field_validator('scenario_type')
    @classmethod
    def validate_scenario_consistency(cls, v: ScenarioType, info) -> ScenarioType:
        """Ensure scenario type matches content."""
        faults = info.data.get('fault_events', [])
        weather = info.data.get('weather')
        cyber = info.data.get('cyber')
        n_minus_k = info.data.get('n_minus_k')
        
        if v == ScenarioType.SINGLE_FAULT and len(faults) > 1:
            raise ValueError("SINGLE_FAULT scenarios must have exactly 1 fault_event")
        
        if v == ScenarioType.EXTREME_WEATHER and weather is None:
            raise ValueError("EXTREME_WEATHER scenarios require weather scenario")
        
        if v == ScenarioType.CYBER_ATTACK and cyber is None:
            raise ValueError("CYBER_ATTACK scenarios require cyber scenario")
        
        if v == ScenarioType.N_MINUS_K and n_minus_k is None:
            raise ValueError("N_MINUS_K scenarios require n_minus_k scenario")
        
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "scenario_id": "hurricane_michael_replay",
                "scenario_type": "composite_scenario",
                "description": "Replay of Hurricane Michael impact on Panama City microgrid (Oct 2018)",
                "fault_events": [
                    {
                        "fault": {
                            "fault_type": "1ph_ground",
                            "magnitude": 0.8,
                            "location_bus": 5,
                            "duration_sec": 120.0,
                            "target_component": "Line_5-7"
                        },
                        "trigger_time_sec": 60.0,
                        "clear_time_sec": 180.0,
                        "event_id": "line_tree_contact"
                    },
                    {
                        "fault": {
                            "fault_type": "gen_trip",
                            "magnitude": 1.0,
                            "location_bus": 3,
                            "target_component": "Solar_Farm_1"
                        },
                        "trigger_time_sec": 90.0,
                        "event_id": "inverter_fault"
                    }
                ],
                "weather": {
                    "severity": "extreme",
                    "wind_peak_mps": 65.0,
                    "precipitation_mm_per_hour": 75.0,
                    "solar_availability_percent": 5.0,
                    "duration_hours": 4.0
                },
                "total_duration_sec": 600.0,
                "tags": ["validation", "hurricane", "historical_event"]
            }
          }
