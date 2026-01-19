"""
Simulation Route Handler.
Orchestrates the real-time interaction between:
1. Data Manager (Context & Session State)
2. PINN Engine (ML Prediction)
3. Physics Core (Safety, Stability, Asset Dynamics)
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
from backend.app.core.pinn_engine import pinn_engine

# Layer 3: Services
from backend.services.data_manager import data_manager
# Try/Except for Financial Engine in case file is missing in specific repo state
try:
    from backend.services.financial_engine import FinancialAnalyzer
    financial_engine = FinancialAnalyzer()
except ImportError:
    class MockFinancial:
        def calculate_metrics(self, *args): return {"cost": 0.0, "co2": 0.0}
    financial_engine = MockFinancial()

# --- SETUP ---
router = APIRouter()
logger = logging.getLogger("SimulationRoute")

# Initialize Local Physics Engines
energy_router = EnergyRouter()
grid_estimator = GridEstimator()
health_model = BatteryHealthEquation()
thermal_model = BatteryThermalEquation(
    mass_kg=250.0, specific_heat_cp=900.0, internal_resistance_r=0.05, 
    heat_transfer_coeff_h=15.0, surface_area_a=2.5
)

class SimulationRequest(BaseModel):
    session_id: str
    tick: int
    load_scaling: float = 1.0
    fault: Optional[FaultPayload] = None

@router.post("/predict")
def run_simulation(req: SimulationRequest):
    """
    Executes one time-step of the Physics-Informed Digital Twin.
    STATEFUL: Updates session memory (SOC, Temp, Diesel) after calculation.
    """
    # 1. RETRIEVE STATEFUL CONTEXT (Data Layer)
    session = data_manager.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please Configure first.")

    # Extract Environment Variables
    weather = session['weather']
    tick = req.tick % 24
    ghi = weather['ghi'][tick]
    temp_amb = weather['temperature'][tick]
    wind = weather['wind_speed'][tick]
    v2g = weather['v2g_availability_kw'][tick]

    # Calculate Load (with Fault Injection)
    base_load_kw = temp_amb * req.load_scaling * 10.0 # Heuristic scaling
    if req.fault and req.fault.is_active:
        base_load_kw *= (1.0 + req.fault.magnitude_pu)
        logger.warning(f"⚠️ Fault Injected: {req.fault.fault_type} ({req.fault.magnitude_pu} pu)")

    # 2. PINN INFERENCE (ML Layer)
    # Get raw voltage prediction from the Neural Network
    # Input Tensor: [Load_PU, Solar_PU, Fault_Mag]
    input_tensor = torch.tensor([[req.load_scaling, ghi/1000.0, 0.0]])
    if req.fault: input_tensor[0, 2] = req.fault.magnitude_pu
    
    raw_volts = pinn_engine.infer_voltage(req.session_id, input_tensor)
    
    # 3. PHYSICS PROJECTION (Constraints Layer)
    # "Hard-clamps Neural Net outputs"
    # Ensure voltage stays within IEEE limits (e.g., 0.9 - 1.1 p.u.)
    # We define limits in settings or default to (0.9, 1.1)
    limits = (0.9, 1.1)
    if hasattr(settings, 'PHYSICS'): limits = (settings.PHYSICS.get("min_voltage_pu", 0.9), 1.1)
    
    safe_volts, violation = project_to_feasible_manifold(raw_volts, limits=limits)
    v_pu = safe_volts.item() if safe_volts.numel() == 1 else safe_volts[0].item()

    # 4. STATE ESTIMATION (Physics Equations)
    # "Calculates Virtual Inertia (H)"
    stability = grid_estimator.forward(
        voltage_angle_rad=torch.tensor([0.1 * req.load_scaling]), 
        active_power_pu=torch.tensor([req.load_scaling])
    )

    # 5. ASSET DYNAMICS (Battery Physics)
    # Calculate Temp & Health updates
    # Estimate current based on load demand vs generation
    net_load = base_load_kw - (ghi + wind * 5.0)
    # If Net Load > 0, Battery Discharges (+Amps). If < 0, Charges (-Amps).
    current_a = (net_load / 0.4) if session.get('battery_soc', 0.5) > 0.2 else 0.0
    current_a = max(-50.0, min(50.0, current_a)) # Clamp to physical max

    # Thermal Update
    prev_temp = session.get('battery_temp_c', 25.0)
    d_temp = thermal_model.forward(
        temp_c=torch.tensor([prev_temp]),
        d_temp_dt=torch.tensor([0.0]), # Simplified
        current_a=torch.tensor([current_a]),
        temp_amb_c=torch.tensor([temp_amb])
    )
    new_temp = prev_temp + d_temp.item()
    
    # SOH Update
    prev_soh = session.get('battery_soh', 1.0)
    new_soh = health_model.forward(
        current_soh=prev_soh,
        temp_c=new_temp,
        current_a=current_a
    )

    # 6. ACTIVE ROUTING (Logic Layer)
    # "The Decision Logic (Diesel vs. Battery)"
    source_state = SourceState(
        solar_kw=ghi,
        wind_kw=wind * 10.0,
        battery_soc=session.get('battery_soc', 0.5), # Use Persistent State
        v2g_available_kw=v2g,
        load_demand_kw=base_load_kw,
        diesel_status=session.get('diesel_status', 'OFF') # Use Persistent State
    )
    
    dispatch = energy_router.compute_dispatch(
        source_state, 
        v_pu
    )

    # 7. STATE PERSISTENCE (Write Back to Session)     # Update SOC
    soc_change = 0.0
    if dispatch['action'] == 'CHARGE_BESS':
        soc_change = 0.05
    elif dispatch['action'] == 'DISCHARGE_BESS':
        soc_change = -0.05
    
    session['battery_soc'] = max(0.0, min(1.0, session.get('battery_soc', 0.5) + soc_change))
    session['battery_temp_c'] = new_temp
    session['battery_soh'] = new_soh
    
    # Update Diesel Status
    if dispatch['action'] == 'START_DIESEL':
        session['diesel_status'] = 'ACTIVE'
    elif dispatch['action'] == 'STOP_DIESEL':
        session['diesel_status'] = 'OFF'

    # 8. FINANCIALS (Services Layer)
    kpis = financial_engine.calculate_metrics(dispatch, base_load_kw, req.tick)

    # 9. CONSTRUCT RESPONSE
    return {
        "grid_state": {
            "avg_voltage_pu": v_pu,
            "stability_index": stability['stability_score'],
            "virtual_inertia_s": stability['virtual_inertia_s'],
            "physics_violation": violation.get('was_violated', False)
        },
        "asset_health": {
            "battery_temp_c": round(new_temp, 2),
            "battery_soh": round(new_soh, 6),
            "battery_soc": round(session['battery_soc'], 4)
        },
        "dispatch": dispatch,
        "kpis": kpis
    }
