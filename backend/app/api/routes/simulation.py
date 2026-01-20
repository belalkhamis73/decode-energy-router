"""
Simulation Route Handler - Enhanced Async Event-Driven Architecture.
Orchestrates real-time interaction between:
1. Data Manager (Context & Session State)
2. PINN Engine (ML Prediction) - PARALLEL EXECUTION
3. Physics Core (Safety, Stability, Asset Dynamics)
4. Energy Router (Decision Logic with Traceability)
5. Financial Engine (Value Calculation)
6. Event Bus (Redis Stream for State Changes)

Map Requirement: "The endpoint handler. Called by: main.py."
"""
from backend.app.core.config import settings
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import torch
import logging
import asyncio
import time
from datetime import datetime
import json

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

# Try/Except for Financial Engine
try:
    from backend.services.financial_engine import FinancialAnalyzer
    financial_engine = FinancialAnalyzer()
except ImportError:
    class MockFinancial:
        def calculate_metrics(self, *args): return {"cost": 0.0, "co2": 0.0}
    financial_engine = MockFinancial()

# Event Bus (Redis Stream Integration)
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

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

# Event Bus Connection Pool
redis_pool = None

async def get_redis():
    """Lazy Redis connection pool initialization."""
    global redis_pool
    if not REDIS_AVAILABLE:
        return None
    if redis_pool is None:
        try:
            redis_pool = await aioredis.create_redis_pool(
                f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}',
                encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            return None
    return redis_pool

async def emit_event(session_id: str, event_type: str, payload: dict):
    """Event-driven state change publisher."""
    redis = await get_redis()
    if redis:
        try:
            event = {
                "timestamp": time.time(),
                "session_id": session_id,
                "type": event_type,
                "payload": payload
            }
            await redis.xadd(
                f"sim_events:{session_id}",
                {"data": json.dumps(event)},
                maxlen=1000
            )
        except Exception as e:
            logger.error(f"Event emission failed: {e}")

# --- REQUEST/RESPONSE MODELS ---
class ParameterOverride(BaseModel):
    """Runtime parameter override."""
    parameter_path: str = Field(..., example="weather.cloud_cover")
    value: float = Field(..., example=0.7)

class SimulationRequest(BaseModel):
    session_id: str
    tick: int
    load_scaling: float = 1.0
    fault: Optional[FaultPayload] = None
    overrides: Optional[List[ParameterOverride]] = None

class ModelPrediction(BaseModel):
    output_kw: float
    efficiency: float
    confidence: float
    inputs_used: Dict[str, float]
    physics_residual: float

class DispatchDecision(BaseModel):
    action: str
    battery_kw: float
    diesel_kw: float
    v2g_kw: float
    curtailment_kw: float
    decision_tree: List[str]
    constraints_active: List[str]

# --- PARALLEL MODEL EXECUTION ---
async def predict_solar_pv(ghi: float, temp: float, cloud_cover: float) -> ModelPrediction:
    """Solar PV model with physics-informed loss."""
    await asyncio.sleep(0.01)  # Simulate async I/O
    efficiency = max(0.0, 0.18 - (temp - 25) * 0.005) * (1 - cloud_cover * 0.7)
    output_kw = ghi * efficiency * 100.0  # 100 m² panel
    confidence = 0.95 if cloud_cover < 0.3 else 0.75
    residual = abs(output_kw - ghi * 0.15 * 100.0) / (ghi + 1e-6)
    
    return ModelPrediction(
        output_kw=output_kw,
        efficiency=efficiency,
        confidence=confidence,
        inputs_used={"ghi": ghi, "temp": temp, "cloud_cover": cloud_cover},
        physics_residual=residual
    )

async def predict_wind_turbine(wind_speed: float, air_density: float) -> ModelPrediction:
    """Wind turbine power curve model."""
    await asyncio.sleep(0.01)
    cut_in, rated, cut_out = 3.5, 12.0, 25.0
    if wind_speed < cut_in or wind_speed > cut_out:
        power = 0.0
    elif wind_speed < rated:
        power = 0.5 * air_density * (wind_speed ** 3) * 0.4  # Cp=0.4
    else:
        power = 50.0  # Rated power
    
    confidence = 0.85 if cut_in < wind_speed < rated else 0.6
    residual = abs(power - wind_speed * 5) / (wind_speed + 1e-6)
    
    return ModelPrediction(
        output_kw=power,
        efficiency=0.4 if wind_speed > cut_in else 0.0,
        confidence=confidence,
        inputs_used={"wind_speed": wind_speed, "air_density": air_density},
        physics_residual=residual
    )

async def predict_battery_thermal(current_a: float, temp_c: float, soc: float) -> ModelPrediction:
    """Battery thermal dynamics prediction."""
    await asyncio.sleep(0.01)
    d_temp = thermal_model.forward(
        torch.tensor([temp_c]),
        torch.tensor([0.0]),
        torch.tensor([current_a]),
        torch.tensor([25.0])
    ).item()
    
    new_temp = temp_c + d_temp
    confidence = 0.9 if abs(current_a) < 30 else 0.7
    residual = abs(d_temp) / (abs(current_a) + 1e-6)
    
    return ModelPrediction(
        output_kw=current_a * 48.0 / 1000.0,  # 48V battery
        efficiency=1.0 - abs(current_a) * 0.05 / 1000.0,
        confidence=confidence,
        inputs_used={"current_a": current_a, "temp_c": temp_c, "soc": soc},
        physics_residual=residual
    )

async def predict_grid_stability(voltage_pu: float, load_pu: float, inertia: float) -> ModelPrediction:
    """Grid stability assessment."""
    await asyncio.sleep(0.01)
    stability = grid_estimator.forward(
        torch.tensor([0.1 * load_pu]),
        torch.tensor([load_pu])
    )
    
    score = stability['stability_score']
    confidence = 0.92 if 0.95 < voltage_pu < 1.05 else 0.6
    
    return ModelPrediction(
        output_kw=0.0,
        efficiency=score,
        confidence=confidence,
        inputs_used={"voltage_pu": voltage_pu, "load_pu": load_pu, "inertia": inertia},
        physics_residual=abs(1.0 - voltage_pu)
    )

async def predict_load_forecast(temp: float, hour: int, base_load: float) -> ModelPrediction:
    """Load forecasting model."""
    await asyncio.sleep(0.01)
    hourly_factor = 1.0 + 0.3 * torch.sin(torch.tensor(hour * 3.14159 / 12)).item()
    temp_factor = 1.0 + (temp - 25) * 0.02
    forecast_kw = base_load * hourly_factor * temp_factor
    
    confidence = 0.88
    residual = abs(forecast_kw - base_load) / (base_load + 1e-6)
    
    return ModelPrediction(
        output_kw=forecast_kw,
        efficiency=1.0,
        confidence=confidence,
        inputs_used={"temp": temp, "hour": hour, "base_load": base_load},
        physics_residual=residual
    )

async def predict_pinn_voltage(session_id: str, input_tensor: torch.Tensor) -> ModelPrediction:
    """PINN voltage prediction wrapper."""
    await asyncio.sleep(0.01)
    raw_volts = pinn_engine.infer_voltage(session_id, input_tensor)
    
    return ModelPrediction(
        output_kw=raw_volts.item() if raw_volts.numel() == 1 else raw_volts[0].item(),
        efficiency=1.0,
        confidence=0.93,
        inputs_used={
            "load_pu": input_tensor[0, 0].item(),
            "solar_pu": input_tensor[0, 1].item(),
            "fault_mag": input_tensor[0, 2].item()
        },
        physics_residual=0.0
    )

# --- CONSTRAINT CHECKING ---
def check_constraints(state: dict, dispatch: dict) -> List[str]:
    """Returns list of active/violated constraints."""
    active = []
    
    # Voltage constraints
    v_pu = state.get('voltage_pu', 1.0)
    if v_pu < 0.95:
        active.append("LOW_VOLTAGE_VIOLATION")
    elif v_pu > 1.05:
        active.append("HIGH_VOLTAGE_VIOLATION")
    
    # Battery SOC constraints
    soc = state.get('battery_soc', 0.5)
    if soc < 0.2:
        active.append("BATTERY_SOC_CRITICAL")
    elif soc > 0.9:
        active.append("BATTERY_SOC_HIGH")
    
    # Thermal constraints
    temp = state.get('battery_temp_c', 25.0)
    if temp > 45.0:
        active.append("BATTERY_OVERTEMP_WARNING")
    elif temp > 55.0:
        active.append("BATTERY_OVERTEMP_CRITICAL")
    
    # Power balance
    gen = dispatch.get('battery_kw', 0) + dispatch.get('diesel_kw', 0)
    load = state.get('load_kw', 0)
    if abs(gen - load) > load * 0.15:
        active.append("POWER_IMBALANCE_WARNING")
    
    return active

# --- ROUTING DECISION TREE ---
def get_routing_explanation(source_state: SourceState, dispatch: dict, v_pu: float) -> List[str]:
    """Returns decision tree trace."""
    tree = []
    
    tree.append(f"STEP 1: Load Demand = {source_state.load_demand_kw:.2f} kW")
    
    renewable_gen = source_state.solar_kw + source_state.wind_kw
    tree.append(f"STEP 2: Renewable Generation = {renewable_gen:.2f} kW")
    
    if renewable_gen >= source_state.load_demand_kw:
        tree.append("DECISION: Renewables sufficient → Check for excess")
        if source_state.battery_soc < 0.9:
            tree.append("ACTION: Charge battery with excess")
        else:
            tree.append("ACTION: Curtail excess generation")
    else:
        deficit = source_state.load_demand_kw - renewable_gen
        tree.append(f"DECISION: Deficit = {deficit:.2f} kW → Check storage")
        
        if source_state.battery_soc > 0.25:
            tree.append("ACTION: Discharge battery")
        elif source_state.v2g_available_kw > 0:
            tree.append("ACTION: Use V2G resources")
        else:
            tree.append("ACTION: Start diesel generator")
    
    if v_pu < 0.95:
        tree.append("CONSTRAINT: Low voltage detected → Increase reactive support")
    
    tree.append(f"FINAL: {dispatch.get('action', 'UNKNOWN')}")
    
    return tree

# --- MAIN SIMULATION ENDPOINT ---
@router.post("/predict")
async def run_simulation(req: SimulationRequest, background_tasks: BackgroundTasks):
    """
    Executes one time-step of the Physics-Informed Digital Twin.
    ASYNC: Parallel model execution with asyncio.gather()
    EVENT-DRIVEN: Emits state changes to Redis stream
    """
    start_time = time.time()
    
    # 1. RETRIEVE STATEFUL CONTEXT
    session = data_manager.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please Configure first.")

    # Apply user overrides
    active_overrides = {}
    if req.overrides:
        for override in req.overrides:
            keys = override.parameter_path.split('.')
            target = session
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = override.value
            active_overrides[override.parameter_path] = override.value

    # Extract environment
    weather = session['weather']
    tick = req.tick % 24
    ghi = weather['ghi'][tick]
    temp_amb = weather['temperature'][tick]
    wind = weather['wind_speed'][tick]
    v2g = weather['v2g_availability_kw'][tick]
    cloud_cover = weather.get('cloud_cover', [0.3] * 24)[tick]

    # Calculate load with fault injection
    base_load_kw = temp_amb * req.load_scaling * 10.0
    if req.fault and req.fault.is_active:
        base_load_kw *= (1.0 + req.fault.magnitude_pu)
        logger.warning(f"⚠️ Fault Injected: {req.fault.fault_type} ({req.fault.magnitude_pu} pu)")
        await emit_event(req.session_id, "FAULT_INJECTED", req.fault.dict())

    # 2. PARALLEL MODEL EXECUTION
    input_tensor = torch.tensor([[req.load_scaling, ghi/1000.0, 0.0]])
    if req.fault:
        input_tensor[0, 2] = req.fault.magnitude_pu

    # Execute all 6 models in parallel
    models_result = await asyncio.gather(
        predict_solar_pv(ghi, temp_amb, cloud_cover),
        predict_wind_turbine(wind, 1.225),
        predict_battery_thermal(
            (base_load_kw - ghi) / 0.4,
            session.get('battery_temp_c', 25.0),
            session.get('battery_soc', 0.5)
        ),
        predict_pinn_voltage(req.session_id, input_tensor),
        predict_grid_stability(1.0, req.load_scaling, 5.0),
        predict_load_forecast(temp_amb, tick, base_load_kw)
    )

    solar_pred, wind_pred, thermal_pred, voltage_pred, stability_pred, load_pred = models_result

    # 3. PHYSICS PROJECTION
    limits = (0.9, 1.1)
    safe_volts, violation = project_to_feasible_manifold(
        torch.tensor([voltage_pred.output_kw]), limits=limits
    )
    v_pu = safe_volts.item()

    # 4. BATTERY HEALTH UPDATE
    current_a = (base_load_kw - (ghi + wind * 5.0)) / 0.4
    current_a = max(-50.0, min(50.0, current_a))
    
    new_temp = session.get('battery_temp_c', 25.0) + thermal_pred.physics_residual * 10
    new_soh = health_model.forward(
        session.get('battery_soh', 1.0),
        new_temp,
        current_a
    )

    # 5. ENERGY ROUTING
    source_state = SourceState(
        solar_kw=solar_pred.output_kw,
        wind_kw=wind_pred.output_kw,
        battery_soc=session.get('battery_soc', 0.5),
        v2g_available_kw=v2g,
        load_demand_kw=base_load_kw,
        diesel_status=session.get('diesel_status', 'OFF')
    )
    
    dispatch = energy_router.compute_dispatch(source_state, v_pu)
    
    # Get routing explanation
    decision_tree = get_routing_explanation(source_state, dispatch, v_pu)
    
    # Check constraints
    current_state = {
        'voltage_pu': v_pu,
        'battery_soc': session.get('battery_soc', 0.5),
        'battery_temp_c': new_temp,
        'load_kw': base_load_kw
    }
    constraints = check_constraints(current_state, dispatch)

    # 6. STATE UPDATES (Event-Driven)
    soc_change = 0.0
    if dispatch['action'] == 'CHARGE_BESS':
        soc_change = 0.05
    elif dispatch['action'] == 'DISCHARGE_BESS':
        soc_change = -0.05
    
    new_soc = max(0.0, min(1.0, session.get('battery_soc', 0.5) + soc_change))
    
    # Emit state change events
    state_changes = {
        'battery_soc': new_soc,
        'battery_temp_c': new_temp,
        'battery_soh': new_soh,
        'diesel_status': 'ACTIVE' if dispatch['action'] == 'START_DIESEL' else session.get('diesel_status', 'OFF')
    }
    
    background_tasks.add_task(emit_event, req.session_id, "STATE_UPDATE", state_changes)
    
    # Update session
    session.update(state_changes)

    # 7. FINANCIALS
    kpis = financial_engine.calculate_metrics(dispatch, base_load_kw, req.tick)

    # 8. NEXT STATE PREDICTION (1-step ahead)
    next_tick = (req.tick + 1) % 24
    next_load = weather['temperature'][next_tick] * req.load_scaling * 10.0

    # 9. CONSTRUCT ENHANCED RESPONSE
    exec_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "timestamp_utc": time.time(),
        "tick_ms": exec_time_ms,
        "models": {
            "solar_pv": solar_pred.dict(),
            "wind_turbine": wind_pred.dict(),
            "battery_thermal": thermal_pred.dict(),
            "pinn_voltage": voltage_pred.dict(),
            "grid_stability": stability_pred.dict(),
            "load_forecast": load_pred.dict()
        },
        "dispatch_decision": {
            "action": dispatch['action'],
            "battery_kw": dispatch.get('battery_kw', 0.0),
            "diesel_kw": dispatch.get('diesel_kw', 0.0),
            "v2g_kw": dispatch.get('v2g_kw', 0.0),
            "curtailment_kw": dispatch.get('curtailment_kw', 0.0),
            "decision_tree": decision_tree,
            "constraints_active": constraints
        },
        "user_overrides_active": active_overrides,
        "next_state_prediction": {
            "tick": next_tick,
            "forecast_load_kw": next_load,
            "forecast_soc": new_soc + soc_change
        },
        "grid_state": {
            "avg_voltage_pu": v_pu,
            "stability_index": stability_pred.efficiency,
            "physics_violation": violation.get('was_violated', False)
        },
        "asset_health": {
            "battery_temp_c": round(new_temp, 2),
            "battery_soh": round(new_soh, 6),
            "battery_soc": round(new_soc, 4)
        },
        "kpis": kpis
    }

# --- NEW ENDPOINTS ---
@router.get("/models/predictions/{session_id}")
async def get_model_predictions(session_id: str, tick: int = 0):
    """Expose raw outputs from all 6 models without routing logic."""
    session = data_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    weather = session['weather']
    tick = tick % 24
    
    models = await asyncio.gather(
        predict_solar_pv(weather['ghi'][tick], weather['temperature'][tick], 0.3),
        predict_wind_turbine(weather['wind_speed'][tick], 1.225),
        predict_battery_thermal(0.0, 25.0, 0.5),
        predict_pinn_voltage(session_id, torch.tensor([[1.0, 0.5, 0.0]])),
        predict_grid_stability(1.0, 1.0, 5.0),
        predict_load_forecast(weather['temperature'][tick], tick, 100.0)
    )
    
    return {
        "solar_pv": models[0].dict(),
        "wind_turbine": models[1].dict(),
        "battery_thermal": models[2].dict(),
        "pinn_voltage": models[3].dict(),
        "grid_stability": models[4].dict(),
        "load_forecast": models[5].dict()
    }

@router.get("/constraints/status/{session_id}")
async def get_constraint_status(session_id: str):
    """Reports active and violated physics constraints."""
    session = data_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = {
        'voltage_pu': 1.0,
        'battery_soc': session.get('battery_soc', 0.5),
        'battery_temp_c': session.get('battery_temp_c', 25.0),
        'load_kw': 100.0
    }
    
    return {
        "constraints": check_constraints(state, {}),
        "state": state
    }

@router.post("/fault/inject")
async def inject_fault(session_id: str, fault: FaultPayload, background_tasks: BackgroundTasks):
    """Runtime fault scenario injection."""
    session = data_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session['active_fault'] = fault.dict()
    background_tasks.add_task(emit_event, session_id, "FAULT_INJECTED", fault.dict())
    
    return {"status": "fault_injected", "fault": fault.dict()}

@router.post("/override/parameter")
async def override_parameter(session_id: str, override: ParameterOverride):
    """Runtime parameter override."""
    session = data_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    keys = override.parameter_path.split('.')
    target = session
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = override.value
    
    return {
        "status": "parameter_overridden",
        "path": override.parameter_pat

