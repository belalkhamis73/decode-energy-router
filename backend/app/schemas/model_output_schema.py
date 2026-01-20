"""
Model Output Schema Definitions.
Standardizes all Digital Twin simulation outputs for consumption by UI, analytics, and storage layers.
Unifies metrics_schema.py with execution metadata and validation results.
Compatible with scenario_schema.py and control_schema.py.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
import sys
sys.path.append('.')
from metrics_schema import SystemMetrics, SystemStatus
from scenario_schema import ScenarioPayload

# --- Enums for Simulation State ---
class SimulationState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SolverStatus(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations_reached"
    DIVERGED = "diverged"
    INFEASIBLE = "infeasible"
    NUMERICAL_ERROR = "numerical_error"

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

# --- Alert/Event Log ---
class AlertEvent(BaseModel):
    """
    Individual alert or notification from simulation.
    Captures constraint violations, warnings, and informational messages.
    """
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when alert was triggered."
    )
    
    severity: AlertSeverity = Field(
        description="Alert priority level."
    )
    
    message: str = Field(
        max_length=500,
        description="Human-readable alert description."
    )
    
    component_id: Optional[str] = Field(
        default=None,
        description="Component that triggered alert (if applicable)."
    )
    
    metric_name: Optional[str] = Field(
        default=None,
        description="Metric that violated threshold (e.g., 'voltage_pu', 'loading_percent')."
    )
    
    measured_value: Optional[float] = Field(
        default=None,
        description="Actual value that triggered alert."
    )
    
    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold that was exceeded."
    )
    
    alert_id: str = Field(
        default="alert_0",
        description="Unique alert identifier."
    )

# --- Solver Diagnostics ---
class SolverDiagnostics(BaseModel):
    """
    Power flow solver convergence information and performance metrics.
    """
    status: SolverStatus = Field(
        description="Final solver state."
    )
    
    iterations_count: int = Field(
        ge=0,
        description="Number of iterations to convergence (or max reached)."
    )
    
    max_mismatch_mw: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="MW - Maximum power balance error at convergence."
    )
    
    solve_time_ms: float = Field(
        ge=0.0,
        description="Milliseconds - Wall-clock time for solver."
    )
    
    algorithm_used: str = Field(
        default="Newton-Raphson",
        description="Solver algorithm (e.g., 'Newton-Raphson', 'Fast Decoupled', 'DC')."
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal solver warnings (e.g., 'Reactive limit hit on Gen_2')."
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="Fatal errors preventing convergence."
    )

# --- Time Series Result ---
class TimeSeriesMetrics(BaseModel):
    """
    Temporal evolution of system metrics over simulation.
    Compact representation for charting/analytics.
    """
    timestamps: List[datetime] = Field(
        description="UTC timestamps for each snapshot."
    )
    
    # Aggregate system metrics (parallel arrays)
    total_generation_mw: List[float] = Field(
        description="MW - Total system generation at each timestep."
    )
    
    total_load_mw: List[float] = Field(
        description="MW - Total system load at each timestep."
    )
    
    average_voltage_pu: List[float] = Field(
        description="p.u. - Mean voltage across all buses."
    )
    
    total_cost_usd: List[float] = Field(
        description="$ - Cumulative operational cost."
    )
    
    carbon_emissions_kg: List[float] = Field(
        description="kg CO₂ - Cumulative carbon footprint."
    )
    
    battery_soc_percent: List[float] = Field(
        description="% - Average battery state of charge."
    )
    
    unserved_energy_mwh: List[float] = Field(
        description="MWh - Cumulative load shed."
    )

    @field_validator('total_generation_mw', 'total_load_mw', 'average_voltage_pu', 
                     'total_cost_usd', 'carbon_emissions_kg', 'battery_soc_percent', 
                     'unserved_energy_mwh')
    @classmethod
    def validate_array_length(cls, v: List[float], info) -> List[float]:
        """Ensure all metric arrays match timestamp array length."""
        timestamps = info.data.get('timestamps', [])
        if len(v) != len(timestamps):
            raise ValueError(f"Metric array length ({len(v)}) must match timestamps ({len(timestamps)})")
        return v

# --- Scenario Result Summary ---
class ScenarioResult(BaseModel):
    """
    High-level summary of scenario outcome.
    Used for quick filtering/comparison of simulation runs.
    """
    scenario_id: str = Field(
        description="Reference to input ScenarioPayload.scenario_id."
    )
    
    simulation_state: SimulationState = Field(
        description="Final simulation execution state."
    )
    
    worst_system_status: SystemStatus = Field(
        description="Most severe system status encountered during simulation."
    )
    
    total_alerts: int = Field(
        default=0,
        ge=0,
        description="Count of all alerts generated."
    )
    
    critical_alerts: int = Field(
        default=0,
        ge=0,
        description="Count of critical/emergency alerts."
    )
    
    convergence_failures: int = Field(
        default=0,
        ge=0,
        description="Number of timesteps where power flow failed to converge."
    )
    
    max_voltage_violation_pu: float = Field(
        default=0.0,
        ge=0.0,
        description="p.u. - Largest voltage deviation from limits (0 = no violations)."
    )
    
    max_line_overload_percent: float = Field(
        default=0.0,
        ge=0.0,
        description="% - Largest line thermal overload (0 = no overloads)."
    )
    
    total_unserved_energy_mwh: float = Field(
        default=0.0,
        ge=0.0,
        description="MWh - Total load shed over simulation."
    )
    
    total_cost_usd: float = Field(
        description="$ - Net operational cost for scenario."
    )
    
    total_carbon_kg: float = Field(
        default=0.0,
        ge=0.0,
        description="kg CO₂ - Total carbon footprint."
    )
    
    passed_validation: bool = Field(
        description="True if scenario met all acceptance criteria."
    )
    
    validation_notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable explanation of validation result."
    )

# --- Master Model Output ---
class ModelOutput(BaseModel):
    """
    Complete Digital Twin simulation output package.
    Combines time-series metrics, alerts, solver diagnostics, and summary results.
    This is the top-level object returned to API clients and stored in databases.
    """
    # --- Identification ---
    output_id: str = Field(
        description="Unique identifier for this simulation run."
    )
    
    scenario_id: str = Field(
        description="Reference to input scenario (ScenarioPayload.scenario_id)."
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when output was generated."
    )
    
    # --- Input Echo (for traceability) ---
    input_scenario: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Copy of input ScenarioPayload for audit trail (optional)."
    )
    
    # --- Execution Metadata ---
    simulation_state: SimulationState = Field(
        description="Final execution state."
    )
    
    execution_time_sec: float = Field(
        ge=0.0,
        description="Seconds - Wall-clock time for entire simulation."
    )
    
    solver_diagnostics: SolverDiagnostics = Field(
        description="Power flow solver performance and convergence info."
    )
    
    # --- Results ---
    final_metrics: SystemMetrics = Field(
        description="System state at end of simulation (t = total_duration_sec)."
    )
    
    time_series: Optional[TimeSeriesMetrics] = Field(
        default=None,
        description="Temporal evolution of key metrics (if time_series output enabled)."
    )
    
    alerts: List[AlertEvent] = Field(
        default_factory=list,
        description="All alerts/events generated during simulation."
    )
    
    summary: ScenarioResult = Field(
        description="High-level outcome summary for quick assessment."
    )
    
    # --- Optional Detailed Snapshots ---
    detailed_snapshots: Optional[List[SystemMetrics]] = Field(
        default=None,
        description="Full SystemMetrics at specific timesteps (for deep analysis)."
    )
    
    # --- Metadata ---
    model_version: str = Field(
        default="v1.0.0",
        description="Digital Twin model version used for simulation."
    )
    
    operator: Optional[str] = Field(
        default=None,
        description="User who initiated simulation."
    )
    
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Operator notes/annotations on simulation results."
    )

    # --- Derived Properties ---
    @property
    def is_successful(self) -> bool:
        """True if simulation completed without fatal errors."""
        return self.simulation_state == SimulationState.COMPLETED

    @property
    def has_violations(self) -> bool:
        """True if any constraint violations occurred."""
        return (
            self.summary.max_voltage_violation_pu > 0 or
            self.summary.max_line_overload_percent > 100.0 or
            self.summary.convergence_failures > 0
        )

    @property
    def reliability_score(self) -> float:
        """
        Normalized reliability metric (0-100).
        100 = perfect (no load shed), 0 = total failure.
        """
        if not self.time_series:
            return 100.0 if self.summary.total_unserved_energy_mwh == 0 else 0.0
        
        total_energy = sum(self.time_series.total_load_mw)
        if total_energy == 0:
            return 100.0
        
        served_energy = total_energy - self.summary.total_unserved_energy_mwh
        return max(0.0, min(100.0, (served_energy / total_energy) * 100.0))

    # --- Validators ---
    @field_validator('alerts')
    @classmethod
    def sort_alerts_by_severity(cls, v: List[AlertEvent]) -> List[AlertEvent]:
        """Sort alerts by severity (emergency first) then timestamp."""
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        return sorted(v, key=lambda a: (severity_order[a.severity], a.timestamp))

    class Config:
        json_schema_extra = {
            "example": {
                "output_id": "sim_20260120_143025_abc123",
                "scenario_id": "hurricane_michael_replay",
                "created_at": "2026-01-20T14:30:25Z",
                "simulation_state": "completed",
                "execution_time_sec": 12.45,
                "solver_diagnostics": {
                    "status": "converged",
                    "iterations_count": 4,
                    "max_mismatch_mw": 0.0001,
                    "solve_time_ms": 150.2,
                    "algorithm_used": "Newton-Raphson"
                },
                "final_metrics": {
                    "timestamp": "2026-01-20T14:40:25Z",
                    "system_status": "warning",
                    "buses": [],
                    "lines": [],
                    "generators": [],
                    "batteries": []
                },
                "alerts": [
                    {
                        "timestamp": "2026-01-20T14:32:00Z",
                        "severity": "critical",
                        "message": "Line_5-7 thermal overload detected",
                        "component_id": "Line_5-7",
                        "metric_name": "loading_percent",
                        "measured_value": 125.5,
                        "threshold_value": 100.0,
                        "alert_id": "alert_001"
                    }
                ],
                "summary": {
                    "scenario_id": "hurricane_michael_replay",
                    "simulation_state": "completed",
                    "worst_system_status": "warning",
                    "total_alerts": 1,
                    "critical_alerts": 1,
                    "convergence_failures": 0,
                    "max_voltage_violation_pu": 0.03,
                    "max_line_overload_percent": 25.5,
                    "total_unserved_energy_mwh": 0.5,
                    "total_cost_usd": 3250.75,
                    "total_carbon_kg": 850.0,
                    "passed_validation": False,
                    "validation_notes": "Line overload exceeded emergency rating"
                },
                "model_version": "v1.2.3",
                "operator": "john.doe@utility.com"
            }
  }
