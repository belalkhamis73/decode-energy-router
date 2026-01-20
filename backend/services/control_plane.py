import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class ParameterType(Enum):
    """Types of controllable parameters"""
    ENVIRONMENTAL = "environmental"
    GRID = "grid"
    LOAD = "load"
    STORAGE = "storage"
    GENERATION = "generation"
    SENSOR = "sensor"

@dataclass
class ParameterConstraint:
    """Physical constraints for a parameter"""
    min_value: float
    max_value: float
    rate_limit: Optional[float] = None  # Max change per second
    unit: str = ""
    description: str = ""

@dataclass
class ScenarioEvent:
    """Single event within a scenario"""
    timestamp: float  # Seconds from scenario start
    parameter: str
    value: Any
    transition_duration: float = 0.0  # Seconds for gradual change

@dataclass
class Scenario:
    """Complete scenario definition"""
    name: str
    description: str
    duration: float  # Total scenario duration in seconds
    events: List[ScenarioEvent]
    initial_conditions: Dict[str, Any]
    tags: List[str]

class ParameterValidator:
    """Ensures user inputs are physically feasible"""
    
    def __init__(self):
        self.constraints = self._initialize_constraints()
    
    def _initialize_constraints(self) -> Dict[str, ParameterConstraint]:
        """Define physical constraints for all parameters"""
        return {
            # Environmental parameters
            "ghi": ParameterConstraint(0, 1500, rate_limit=100, unit="W/m²", 
                                      description="Global Horizontal Irradiance"),
            "ambient_temp": ParameterConstraint(-40, 60, rate_limit=5, unit="°C",
                                               description="Ambient temperature"),
            "wind_speed": ParameterConstraint(0, 70, rate_limit=10, unit="m/s",
                                             description="Wind speed"),
            "humidity": ParameterConstraint(0, 100, unit="%",
                                          description="Relative humidity"),
            "pressure": ParameterConstraint(800, 1100, unit="hPa",
                                          description="Atmospheric pressure"),
            
            # Grid parameters
            "grid_voltage": ParameterConstraint(0.85, 1.15, rate_limit=0.1, unit="p.u.",
                                               description="Grid voltage (per unit)"),
            "grid_frequency": ParameterConstraint(59.5, 60.5, rate_limit=0.5, unit="Hz",
                                                 description="Grid frequency"),
            "grid_available": ParameterConstraint(0, 1, unit="bool",
                                                 description="Grid connection status"),
            
            # Load parameters
            "base_load": ParameterConstraint(0, 10000, rate_limit=500, unit="kW",
                                           description="Base facility load"),
            "ev_load": ParameterConstraint(0, 5000, rate_limit=1000, unit="kW",
                                         description="EV charging load"),
            "critical_load": ParameterConstraint(0, 2000, unit="kW",
                                               description="Critical load requirement"),
            
            # Storage parameters
            "battery_soc": ParameterConstraint(0, 100, unit="%",
                                              description="Battery state of charge"),
            "battery_temp": ParameterConstraint(-20, 60, rate_limit=2, unit="°C",
                                               description="Battery temperature"),
            
            # Generation parameters
            "diesel_available": ParameterConstraint(0, 1, unit="bool",
                                                   description="Diesel generator availability"),
            "diesel_fuel": ParameterConstraint(0, 100, unit="%",
                                              description="Diesel fuel level"),
            
            # Sensor parameters
            "sensor_noise": ParameterConstraint(0, 10, unit="%",
                                               description="Sensor noise level"),
            "comms_latency": ParameterConstraint(0, 5000, unit="ms",
                                                description="Communication latency"),
            "comms_packet_loss": ParameterConstraint(0, 100, unit="%",
                                                    description="Packet loss rate"),
        }
    
    def validate_override(self, parameter: str, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a parameter override
        
        Returns:
            (is_valid, error_message)
        """
        if parameter not in self.constraints:
            return False, f"Unknown parameter: {parameter}"
        
        constraint = self.constraints[parameter]
        
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return False, f"Invalid value type for {parameter}: {value}"
        
        if numeric_value < constraint.min_value:
            return False, f"{parameter} below minimum: {numeric_value} < {constraint.min_value} {constraint.unit}"
        
        if numeric_value > constraint.max_value:
            return False, f"{parameter} above maximum: {numeric_value} > {constraint.max_value} {constraint.unit}"
        
        return True, None
    
    def validate_rate(self, parameter: str, current_value: float, 
                     new_value: float, time_delta: float) -> Tuple[bool, Optional[str]]:
        """Validate rate of change constraint"""
        if parameter not in self.constraints:
            return True, None
        
        constraint = self.constraints[parameter]
        if constraint.rate_limit is None:
            return True, None
        
        if time_delta <= 0:
            return False, "Time delta must be positive"
        
        rate = abs(new_value - current_value) / time_delta
        
        if rate > constraint.rate_limit:
            return False, (f"{parameter} rate too high: {rate:.2f} {constraint.unit}/s "
                          f"> {constraint.rate_limit} {constraint.unit}/s")
        
        return True, None

class ScenarioLibrary:
    """Predefined test scenarios"""
    
    def __init__(self):
        self.scenarios = self._initialize_scenarios()
    
    def _initialize_scenarios(self) -> Dict[str, Scenario]:
        """Create predefined scenario templates"""
        scenarios = {}
        
        # Solar Eclipse Scenario
        eclipse_events = []
        eclipse_duration = 1800  # 30 minutes
        for t in range(0, eclipse_duration, 60):
            # Gradual reduction to 0, then recovery
            if t < eclipse_duration / 2:
                ghi = 1000 * (1 - 2 * t / eclipse_duration)
            else:
                ghi = 1000 * (2 * t / eclipse_duration - 1)
            eclipse_events.append(ScenarioEvent(t, "ghi", max(0, ghi), 60))
        
        scenarios["solar_eclipse"] = Scenario(
            name="Solar Eclipse",
            description="Gradual GHI reduction to 0 over 15 minutes, then recovery",
            duration=eclipse_duration,
            events=eclipse_events,
            initial_conditions={"ghi": 1000},
            tags=["environmental", "generation", "critical"]
        )
        
        # Hurricane Scenario
        scenarios["hurricane"] = Scenario(
            name="Hurricane",
            description="Wind speed ramp to turbine cut-out, followed by grid outage",
            duration=3600,
            events=[
                ScenarioEvent(0, "wind_speed", 15, 300),
                ScenarioEvent(300, "wind_speed", 25, 300),  # Approaching cut-out
                ScenarioEvent(600, "wind_speed", 30, 60),   # Turbine shutdown
                ScenarioEvent(900, "grid_voltage", 0.95, 60),
                ScenarioEvent(1200, "grid_voltage", 0.85, 30),
                ScenarioEvent(1500, "grid_available", 0, 1),  # Grid loss
                ScenarioEvent(3000, "wind_speed", 35, 0),
            ],
            initial_conditions={"wind_speed": 10, "grid_available": 1},
            tags=["environmental", "grid", "emergency"]
        )
        
        # Heat Wave Scenario
        scenarios["heat_wave"] = Scenario(
            name="Heat Wave",
            description="Elevated ambient temperature causing battery thermal stress",
            duration=14400,  # 4 hours
            events=[
                ScenarioEvent(0, "ambient_temp", 35, 1800),
                ScenarioEvent(1800, "ambient_temp", 42, 3600),
                ScenarioEvent(5400, "ambient_temp", 45, 1800),
                ScenarioEvent(7200, "battery_temp", 50, 1800),
                ScenarioEvent(10800, "ambient_temp", 38, 3600),
            ],
            initial_conditions={"ambient_temp": 25, "battery_temp": 25},
            tags=["environmental", "storage", "thermal"]
        )
        
        # Cyber Attack Scenario
        scenarios["cyber_attack"] = Scenario(
            name="Cyber Attack",
            description="Simulated sensor noise and communication loss",
            duration=1800,
            events=[
                ScenarioEvent(0, "sensor_noise", 0.5, 10),
                ScenarioEvent(300, "sensor_noise", 2.0, 30),
                ScenarioEvent(600, "comms_latency", 500, 60),
                ScenarioEvent(900, "comms_packet_loss", 15, 30),
                ScenarioEvent(1200, "sensor_noise", 5.0, 10),
                ScenarioEvent(1200, "comms_packet_loss", 40, 10),
                ScenarioEvent(1500, "sensor_noise", 1.0, 60),
                ScenarioEvent(1600, "comms_latency", 100, 120),
                ScenarioEvent(1700, "comms_packet_loss", 5, 100),
            ],
            initial_conditions={"sensor_noise": 0, "comms_latency": 50, "comms_packet_loss": 0},
            tags=["cybersecurity", "sensor", "communications"]
        )
        
        # Generator Failure Scenario
        scenarios["generator_failure"] = Scenario(
            name="Generator Failure",
            description="Diesel generator trip while under load",
            duration=600,
            events=[
                ScenarioEvent(0, "grid_available", 0, 1),  # Grid already out
                ScenarioEvent(60, "base_load", 800, 30),
                ScenarioEvent(180, "diesel_available", 0, 1),  # Sudden trip
                ScenarioEvent(300, "battery_soc", 70, 0),  # Battery takes over
            ],
            initial_conditions={
                "grid_available": 0,
                "diesel_available": 1,
                "base_load": 500,
                "battery_soc": 85
            },
            tags=["generation", "emergency", "backup"]
        )
        
        # Cloud Cover Transient (Gorilla Pattern)
        cloud_events = []
        for i in range(0, 600, 30):
            # Random fluctuations between 200-1000 W/m²
            ghi = 600 + 400 * np.sin(i / 50) + 200 * np.random.random()
            cloud_events.append(ScenarioEvent(i, "ghi", ghi, 30))
        
        scenarios["cloud_transient"] = Scenario(
            name="Cloud Cover Transient",
            description="Rapid GHI fluctuations simulating passing clouds",
            duration=600,
            events=cloud_events,
            initial_conditions={"ghi": 1000},
            tags=["environmental", "generation", "variability"]
        )
        
        # EV Fleet Event
        scenarios["ev_fleet_event"] = Scenario(
            name="EV Fleet Event",
            description="Mass V2G connection and disconnection",
            duration=1200,
            events=[
                ScenarioEvent(0, "ev_load", 0, 0),
                ScenarioEvent(60, "ev_load", 500, 120),   # Fleet arrives
                ScenarioEvent(300, "ev_load", 1500, 180), # Peak charging
                ScenarioEvent(600, "ev_load", 2000, 60),  # V2G begins
                ScenarioEvent(900, "ev_load", -500, 120), # V2G discharge
                ScenarioEvent(1080, "ev_load", 0, 60),    # Fleet departs
            ],
            initial_conditions={"ev_load": 0},
            tags=["load", "v2g", "storage"]
        )
        
        return scenarios
    
    def get_scenario(self, name: str) -> Optional[Scenario]:
        """Retrieve scenario by name"""
        return self.scenarios.get(name)
    
    def list_scenarios(self, tags: Optional[List[str]] = None) -> List[str]:
        """List available scenarios, optionally filtered by tags"""
        if tags is None:
            return list(self.scenarios.keys())
        
        return [
            name for name, scenario in self.scenarios.items()
            if any(tag in scenario.tags for tag in tags)
        ]

class ControlPlane:
    """Main control surface manager"""
    
    def __init__(self):
        self.validator = ParameterValidator()
        self.scenario_library = ScenarioLibrary()
        self.active_overrides: Dict[str, Any] = {}
        self.active_scenario: Optional[str] = None
        self.scenario_start_time: Optional[datetime] = None
        self.scenario_history: List[Dict[str, Any]] = []
        self.current_values: Dict[str, float] = {}
    
    def get_control_schema(self) -> Dict[str, Any]:
        """Returns JSON schema of all controllable parameters"""
        schema = {
            "parameters": {},
            "scenarios": {},
            "metadata": {
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add parameter definitions
        for param_name, constraint in self.validator.constraints.items():
            schema["parameters"][param_name] = {
                "min": constraint.min_value,
                "max": constraint.max_value,
                "rate_limit": constraint.rate_limit,
                "unit": constraint.unit,
                "description": constraint.description,
                "current_value": self.active_overrides.get(param_name)
            }
        
        # Add scenario definitions
        for scenario_name in self.scenario_library.list_scenarios():
            scenario = self.scenario_library.get_scenario(scenario_name)
            schema["scenarios"][scenario_name] = {
                "description": scenario.description,
                "duration": scenario.duration,
                "tags": scenario.tags,
                "event_count": len(scenario.events)
            }
        
        return schema
    
    def validate_override(self, parameter: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Checks physical constraints"""
        return self.validator.validate_override(parameter, value)
    
    def apply_bulk_overrides(self, overrides_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Batch parameter updates
        
        Returns:
            Dict with success status and any errors
        """
        results = {
            "success": True,
            "applied": [],
            "failed": [],
            "errors": []
        }
        
        for parameter, value in overrides_dict.items():
            is_valid, error = self.validate_override(parameter, value)
            
            if is_valid:
                self.active_overrides[parameter] = value
                self.current_values[parameter] = float(value)
                results["applied"].append(parameter)
            else:
                results["success"] = False
                results["failed"].append(parameter)
                results["errors"].append({"parameter": parameter, "error": error})
        
        return results
    
    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Apply predefined scenario template"""
        scenario = self.scenario_library.get_scenario(scenario_name)
        
        if scenario is None:
            return {
                "success": False,
                "error": f"Scenario '{scenario_name}' not found"
            }
        
        # Apply initial conditions
        self.apply_bulk_overrides(scenario.initial_conditions)
        
        # Record scenario activation
        self.active_scenario = scenario_name
        self.scenario_start_time = datetime.now()
        
        self.scenario_history.append({
            "scenario": scenario_name,
            "start_time": self.scenario_start_time.isoformat(),
            "duration": scenario.duration,
            "initial_conditions": scenario.initial_conditions
        })
        
        return {
            "success": True,
            "scenario": scenario_name,
            "duration": scenario.duration,
            "events": len(scenario.events),
            "start_time": self.scenario_start_time.isoformat()
        }
    
    def create_custom_scenario(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """User-defined scenario creation"""
        required_fields = ["name", "description", "duration", "events"]
        
        # Validate required fields
        for field in required_fields:
            if field not in params:
                return {
                    "success": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Convert event dictionaries to ScenarioEvent objects
        try:
            events = [
                ScenarioEvent(
                    timestamp=e["timestamp"],
                    parameter=e["parameter"],
                    value=e["value"],
                    transition_duration=e.get("transition_duration", 0.0)
                )
                for e in params["events"]
            ]
        except KeyError as e:
            return {
                "success": False,
                "error": f"Invalid event structure: missing {str(e)}"
            }
        
        # Validate all events
        for event in events:
            is_valid, error = self.validate_override(event.parameter, event.value)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Event validation failed: {error}"
                }
        
        # Create scenario
        scenario = Scenario(
            name=params["name"],
            description=params["description"],
            duration=params["duration"],
            events=events,
            initial_conditions=params.get("initial_conditions", {}),
            tags=params.get("tags", ["custom"])
        )
        
        # Add to library
        self.scenario_library.scenarios[params["name"]] = scenario
        
        return {
            "success": True,
            "scenario": params["name"],
            "events": len(events)
        }
    
    def get_scenario_history(self) -> List[Dict[str, Any]]:
        """Log of all applied scenarios"""
        return self.scenario_history
    
    def get_active_overrides(self) -> Dict[str, Any]:
        """Get currently active parameter overrides"""
        return self.active_overrides.copy()
    
    def clear_overrides(self) -> None:
        """Clear all active overrides"""
        self.active_overrides.clear()
        self.active_scenario = None
        self.scenario_start_time = None
    
    def get_scenario_progress(self) -> Optional[Dict[str, Any]]:
        """Get progress of active scenario"""
        if self.active_scenario is None or self.scenario_start_time is None:
            return None
        
        scenario = self.scenario_library.get_scenario(self.active_scenario)
        if scenario is None:
            return None
        
        elapsed = (datetime.now() - self.scenario_start_time).total_seconds()
        progress = min(100, (elapsed / scenario.duration) * 100)
        
        return {
            "scenario": self.active_scenario,
            "elapsed_seconds": elapsed,
            "total_duration": scenario.duration,
            "progress_percent": progress,
            "completed": elapsed >= scenario.duration
        }
    
    def update_scenario_state(self, current_time: float) -> Dict[str, Any]:
        """
        Update parameter values based on active scenario timeline
        
        Args:
            current_time: Seconds since scenario start
            
        Returns:
            Dict of parameter updates to apply
        """
        if self.active_scenario is None:
            return {}
        
        scenario = self.scenario_library.get_scenario(self.active_scenario)
        if scenario is None:
            return {}
        
        updates = {}
        
        # Find events that should be active at current_time
        for event in scenario.events:
            if event.timestamp <= current_time < event.timestamp + event.transition_duration:
                # Interpolate value during transition
                progress = (current_time - event.timestamp) / event.transition_duration
                if event.parameter in self.current_values:
                    old_value = self.current_values[event.parameter]
                    new_value = old_value + (event.value - old_value) * progress
                    updates[event.parameter] = new_value
            elif current_time >= event.timestamp + event.transition_duration:
                # Use final value
                updates[event.parameter] = event.value
        
        return updates
