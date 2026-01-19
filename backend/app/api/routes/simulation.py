"""
Simulation Route Handler.
Orchestrates the real-time interaction between:
1. Data Manager (Context)
2. PINN Engine (ML Prediction)
3. Physics Core (Safety & Stability)
4. Energy Router (Decision Logic)
5. Financial Engine (Value Calculation)

Map Requirement: "The endpoint handler. Called by: main.py."
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import torch
import logging

# --- IMPORT LAYERS ---
# Layer 0: Schemas & Config
from backend.app.schemas.fault_payload import FaultPayload
from backend.app.core.config import settings

# Layer 1: Physics Core
from physics_core.equations.energy_router import EnergyRouter, SourceState
from physics_core.equations.grid_estimator import GridEstimator
from physics_core.constraints.projection import project_to_feasible_manifold
from physics_core.equations.battery_health import BatteryHealthEquation
from physics_core.equations.battery_thermal import BatteryThermalEquation

# Layer 2: ML Engine
from backend.core.pinn_engine import pinn_engine

# Layer 3: Services
from backend.services.data_manager import data_manager
from backend.services.financial_engine import FinancialAnalyzer

# --- SETUP ---
router = APIRouter()
logger = logging.getLogger("SimulationRoute")

# Initialize Local Logic Engines
energy_router = EnergyRouter()
grid_estimator = GridEstimator()
financial_engine = FinancialAnalyzer()
health_model = BatteryHealthEquation()
thermal_model = BatteryThermalEquation(
    mass_kg=250.0, specific_heat_cp=900.0, internal_resistance_r=0.05, 
    heat_transfer_coeff_h=15.0, surface_area_a=2.5
)

# Mock Session Store Access (In production, use Redis via Dependency Injection)
# We assume main.py injects the session state or we access a shared store
# For this modular design, we'll accept session state as a dependency or argument if possible,
# but to keep the signature clean for FastAPI, we'll access a global store reference or mock it.
# Ideally, session state should be managed by a Service. 
# Here we will simulate state retrieval for the "Stateless" route design.

class SimulationRequest(BaseModel):
    session_id: str
    tick: int
    load_scaling: float
    fault: Optional[FaultPayload] = None

@router.post("/predict")
async def run_simulation(req: SimulationRequest):
    """
    Executes one time-step of the Physics-Informed Digital Twin.
    """
    # 1. RETRIEVE CONTEXT (Data Layer)
    # We assume the session was configured previously.
    # For now, we fetch standard profiles directly from DataManager.
    # In a real app, retrieve 'weather' and 'topology' from Redis using req.session_id.
    weather = data_manager.get_weather_profile("solar_egypt") 
    
    # Validation
    if not pinn_engine.is_ready:
        # If model isn't trained, we can either error out or run in "Physics-Only" mode.
        # We choose Physics-Only mode for robustness.
        logger.warning(f"Session {req.session_id}: Running in Physics-Only mode (Model not ready)")

    # 2. PINN INFERENCE (ML Layer)
    # Get raw voltage prediction from the Neural Network
    raw_volts = pinn_engine.infer_voltage(req.load_scaling, weather, req.tick)
    
    # 3. PHYSICS PROJECTION (Constraints Layer)
    # "Hard-clamps Neural Net outputs"
    # Ensure voltage stays within IEEE limits (e.g., 0.9 - 1.1 p.u.)
    safe_volts, violation = project_to_feasible_manifold(
        raw_volts, 
        limits=(settings.PHYSICS.get("min_voltage_pu", 0.9), 1.1)
    )
    v_pu = safe_volts.item()

    # 4. STATE ESTIMATION (Physics Equations)
    # "Calculates Virtual Inertia (H)"
    # We estimate grid stiffness based on the load
    stability = grid_estimator.forward(
        voltage_angle_rad=torch.tensor([0.1 * req.load_scaling]), 
        active_power_pu=torch.tensor([req.load_scaling])
    )

    # 5. ASSET DYNAMICS (Battery Physics)
    # Calculate Temp & Health
    # Mock current based on load (High load -> Discharge)
    base_load_kw = req.load_scaling * 500.0
    current_a = 50.0 if base_load_kw > 400 else -20.0
    
    # Thermal Update
    # Note: In a stateless route, we'd fetch prev_temp from DB. 
    # Here we assume a stable 25C start for the step calc.
    d_temp = thermal_model.forward(
        current_a=torch.tensor([current_a]),
        temp_amb_c=torch.tensor([weather['temperature'][req.tick]]),
        d_temp_dt=torch.tensor([0.0])
    )
    new_temp = 25.0 + d_temp.item() # Delta from ambient
    
    # SOH Update
    new_soh = health_model.forward(
        current_soh=1.0, # Mock prev SOH
        temp_c=new_temp,
        current_a=current_a
    )

    # 6. ACTIVE ROUTING (Logic Layer)
    # "The Decision Logic (Diesel vs. Battery)"
    source_state = SourceState(
        solar_kw=weather['ghi'][req.tick],
        wind_kw=weather['wind_speed'][req.tick] * 5.0,
        battery_soc=0.8, # Mock SOC
        v2g_available_kw=weather['v2g_availability_kw'][req.tick],
        load_demand_kw=base_load_kw,
        diesel_status="OFF"
    )
    
    dispatch = energy_router.compute_dispatch(
        source_state, 
        v_pu, 
        stability_score=stability['stability_score']
    )

    # 7. FINANCIALS (Services Layer)
    # "Calculates $$$ and CO2"
    kpis = financial_engine.calculate_metrics(dispatch, base_load_kw, req.tick)

    # 8. CONSTRUCT RESPONSE
    return {
        "grid_state": {
            "voltage_pu": v_pu,
            "stability_index": stability['stability_score'],
            "virtual_inertia": stability['virtual_inertia_s'],
            "physics_violation": violation.get('was_violated', False)
        },
        "asset_health": {
            "battery_temp_c": round(new_temp, 2),
            "battery_soh": round(new_soh, 6)
        },
        "dispatch": dispatch,
        "kpis": kpis
    }
