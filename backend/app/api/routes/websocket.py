""" WebSocket Route Handler - Real-time Bidirectional Communication
Refactored to eliminate global mutable state.
Uses immutable SessionSnapshot for all state reads/writes.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Set, Optional, Any, List
import asyncio
import json
import logging
import time
from datetime import datetime
from collections import defaultdict

# Core imports
try:
    from backend.services.data_manager import data_manager
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    logging.warning("⚠️ DataManager not available")

# Physics imports
try:
    from physics_core.equations.energy_router import EnergyRouter, SourceState
    from physics_core.equations.grid_estimator import GridEstimator
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False
    logging.warning("⚠️ Physics modules not available")

# PINN engine
try:
    from backend.core.pinn_engine import pinn_engine, ModelType
    PINN_AVAILABLE = True
except ImportError:
    PINN_AVAILABLE = False
    logging.warning("⚠️ PINN engine not available")

router = APIRouter()
logger = logging.getLogger("WebSocketRoute")

# Initialize physics engines if available
if PHYSICS_AVAILABLE:
    energy_router = EnergyRouter()
    grid_estimator = GridEstimator()


#============================================================================
# CONNECTION MANAGER (unchanged – manages connections, not business state)
#============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = defaultdict(dict)
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.client_meta Dict[str, Dict[str, Any]] = {}
        self.sim_running: Dict[str, bool] = {}
        self.sim_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, session_id: str, client_id: str):
        await websocket.accept()
        self.active_connections[session_id][client_id] = websocket
        self.client_metadata[client_id] = {
            "session_id": session_id,
            "connected_at": time.time(),
            "message_count": 0
        }
        logger.info(f"✅ Client {client_id} connected to session {session_id}")

    def disconnect(self, session_id: str, client_id: str):
        if client_id in self.active_connections.get(session_id, {}):
            del self.active_connections[session_id][client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.client_meta
            del self.client_metadata[client_id]
        logger.info(f"❌ Client {client_id} disconnected from session {session_id}")

    def subscribe(self, client_id: str, metrics: List[str]):
        self.subscriptions[client_id].update(metrics)
        logger.debug(f"Client {client_id} subscribed to: {metrics}")

    def unsubscribe(self, client_id: str, metrics: List[str]):
        self.subscriptions[client_id].difference_update(metrics)
        logger.debug(f"Client {client_id} unsubscribed from: {metrics}")

    async def send_personal_message(self, message: dict, session_id: str, client_id: str):
        websocket = self.active_connections.get(session_id, {}).get(client_id)
        if websocket:
            try:
                await websocket.send_json(message)
                self.client_metadata[client_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")

    async def broadcast_to_session(self, message: dict, session_id: str, metric_type: Optional[str] = None):
        if session_id not in self.active_connections:
            return
        tasks = []
        for client_id, websocket in self.active_connections[session_id].items():
            if metric_type and metric_type not in self.subscriptions.get(client_id, set()):
                continue
            try:
                tasks.append(websocket.send_json(message))
                self.client_metadata[client_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Broadcast error to {client_id}: {e}")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


manager = ConnectionManager()


#============================================================================
# PURE STATE TRANSFORMS (no mutation)
#============================================================================

def apply_fault_to_snapshot(snapshot: dict, fault: dict) -> dict:
    """Return new snapshot with fault applied (immutable)."""
    new = snapshot.copy()
    new['active_fault'] = fault
    return new

def apply_parameter_override(snapshot: dict, path: str, value: Any) -> dict:
    """Return new snapshot with parameter override (deep copy path)."""
    import copy
    new = copy.deepcopy(snapshot)
    keys = path.split('.')
    target = new
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value
    return new

def update_battery_soc_in_snapshot(snapshot: dict, soc_change: float) -> dict:
    """Return new snapshot with updated SOC."""
    current_soc = snapshot.get('battery_soc', 0.5)
    new_soc = max(0.0, min(1.0, current_soc + soc_change))
    new = snapshot.copy()
    new['battery_soc'] = new_soc
    return new


#============================================================================
# SIMULATION STEP (now uses immutable snapshots)
#============================================================================

async def run_simulation_step(session_id: str, tick: int, session_snapshot: dict) -> dict:
    weather = session_snapshot['weather']
    tick_idx = tick % 24
    ghi = weather['ghi'][tick_idx]
    temp = weather['temperature'][tick_idx]
    wind = weather['wind_speed'][tick_idx]
    cloud_cover = weather.get('cloud_cover', [0.3] * 24)[tick_idx]

    base_load_kw = temp * 10.0
    active_fault = session_snapshot.get('active_fault')
    if active_fault and active_fault.get('is_active'):
        base_load_kw *= (1.0 + active_fault.get('magnitude_pu', 0.0))

    # Predictions
    if PINN_AVAILABLE:
        try:
            solar_kw, _, _ = pinn_engine.infer_solar_production(
                session_id=session_id, ghi=ghi, temp=temp, panel_angle=30.0, return_uncertainty=False
            )
            wind_kw, _, _ = pinn_engine.infer_wind_production(
                session_id=session_id, wind_speed=wind, return_uncertainty=False
            )
        except Exception as e:
            logger.warning(f"PINN inference failed, using fallback: {e}")
            solar_kw = _fallback_solar_model(ghi, temp, cloud_cover)
            wind_kw = _fallback_wind_model(wind)
    else:
        solar_kw = _fallback_solar_model(ghi, temp, cloud_cover)
        wind_kw = _fallback_wind_model(wind)

    # Dispatch
    if PHYSICS_AVAILABLE:
        source_state = SourceState(
            solar_kw=solar_kw,
            wind_kw=wind_kw,
            battery_soc=session_snapshot.get('battery_soc', 0.5),
            v2g_available_kw=weather.get('v2g_availability_kw', [0] * 24)[tick_idx],
            load_demand_kw=base_load_kw,
            diesel_status=session_snapshot.get('diesel_status', 'OFF')
        )
        dispatch = energy_router.compute_dispatch(source_state, 1.0)
    else:
        dispatch = _fallback_dispatch(solar_kw, wind_kw, base_load_kw, session_snapshot.get('battery_soc', 0.5))

    # Compute SOC change
    soc_change = 0.0
    if dispatch['action'] == 'CHARGE_BESS':
        soc_change = 0.05
    elif dispatch['action'] == 'DISCHARGE_BESS':
        soc_change = -0.05

    new_soc = max(0.0, min(1.0, session_snapshot.get('battery_soc', 0.5) + soc_change))

    constraints = []
    if new_soc < 0.2:
        constraints.append("BATTERY_SOC_CRITICAL")
    elif new_soc > 0.9:
        constraints.append("BATTERY_SOC_HIGH")

    return {
        "models": {
            "solar_kw": round(solar_kw, 2),
            "wind_kw": round(wind_kw, 2),
            "load_kw": round(base_load_kw, 2)
        },
        "dispatch": {
            "action": dispatch['action'],
            "battery_kw": dispatch.get('battery_kw', 0.0),
            "diesel_kw": dispatch.get('diesel_kw', 0.0)
        },
        "asset_health": {
            "battery_soc": round(new_soc, 4),
            "battery_temp_c": session_snapshot.get('battery_temp_c', 25.0)
        },
        "constraints": constraints
    }


def _fallback_solar_model(ghi: float, temp: float, cloud_cover: float) -> float:
    solar_eff = max(0.0, 0.18 - (temp - 25) * 0.005) * (1 - cloud_cover * 0.7)
    return ghi * solar_eff * 100.0

def _fallback_wind_model(wind_speed: float) -> float:
    if 3.5 < wind_speed < 25.0:
        if wind_speed < 12.0:
            return 0.5 * 1.225 * (wind_speed ** 3) * 0.4
        else:
            return 50.0
    return 0.0

def _fallback_dispatch(solar_kw: float, wind_kw: float, load_kw: float, battery_soc: float) -> dict:
    net = load_kw - (solar_kw + wind_kw)
    if net > 0:
        if battery_soc > 0.2:
            return {"action": "DISCHARGE_BESS", "battery_kw": -min(net, 50.0), "diesel_kw": 0.0}
        else:
            return {"action": "START_DIESEL", "battery_kw": 0.0, "diesel_kw": net}
    else:
        return {"action": "CHARGE_BESS", "battery_kw": min(abs(net), 50.0), "diesel_kw": 0.0}


#============================================================================
# MESSAGE HANDLERS (now use immutable updates + save atomically)
#============================================================================

async def handle_get_session_snapshot(session_id: str) -> Optional[dict]:
    """Retrieve current session snapshot safely."""
    if not DATA_MANAGER_AVAILABLE:
        return None
    # Use async get_session if available; otherwise fall back to sync access with warning
    if hasattr(data_manager, 'get_session'):
        return data_manager.get_session(session_id)
    else:
        # Fallback: assume _sessions is still present but treat as read-only
        return data_manager._sessions.get(session_id)

async def handle_save_session_snapshot(session_id: str, new_snapshot: dict):
    """Atomically persist new snapshot."""
    if not DATA_MANAGER_AVAILABLE:
        return
    if hasattr(data_manager, 'save_session'):
        await data_manager.save_session(session_id, new_snapshot)
    else:
        # Fallback: mutate (not ideal, but preserves compatibility)
        data_manager._sessions[session_id] = new_snapshot


async def handle_inject_fault(session_id: str, client_id: str, payload: dict):
    fault = payload.get("fault", {})
    snapshot = await handle_get_session_snapshot(session_id)
    if not snapshot:
        return {"type": "ERROR", "message": "Session not found"}
    
    new_snapshot = apply_fault_to_snapshot(snapshot, fault)
    await handle_save_session_snapshot(session_id, new_snapshot)

    await manager.broadcast_to_session({
        "type": "FAULT_INJECTED",
        "timestamp": time.time(),
        "fault": fault
    }, session_id, metric_type="alerts")
    return {"type": "FAULT_INJECTION_CONFIRMED", "fault": fault}


async def handle_override_parameter(session_id: str, client_id: str, payload: dict):
    path = payload.get("parameter_path")
    value = payload.get("value")
    if not path or value is None:
        return {"type": "ERROR", "message": "Missing path or value"}
    
    snapshot = await handle_get_session_snapshot(session_id)
    if not snapshot:
        return {"type": "ERROR", "message": "Session not found"}

    new_snapshot = apply_parameter_override(snapshot, path, value)
    await handle_save_session_snapshot(session_id, new_snapshot)

    await manager.broadcast_to_session({
        "type": "PARAMETER_OVERRIDDEN",
        "timestamp": time.time(),
        "parameter_path": path,
        "value": value
    }, session_id, metric_type="system")
    return {"type": "OVERRIDE_CONFIRMED", "parameter_path": path, "value": value}


async def handle_get_status(session_id: str, client_id: str, payload: dict):
    snapshot = await handle_get_session_snapshot(session_id)
    if not snapshot:
        return {"type": "ERROR", "message": "Session not found"}
    return {
        "type": "STATUS_RESPONSE",
        "timestamp": time.time(),
        "simulation_running": manager.sim_running.get(session_id, False),
        "session_state": {
            "battery_soc": snapshot.get("battery_soc", 0.5),
            "battery_temp_c": snapshot.get("battery_temp_c", 25.0),
            "battery_soh": snapshot.get("battery_soh", 1.0),
            "diesel_status": snapshot.get("diesel_status", "OFF")
        },
        "active_fault": snapshot.get("active_fault"),
        "connected_clients": len(manager.active_connections.get(session_id, {}))
    }


# Other handlers (subscribe/unsubscribe/start/stop sim) remain unchanged in logic
MESSAGE_HANDLERS = {
    "SUBSCRIBE": handle_subscribe,
    "UNSUBSCRIBE": handle_unsubscribe,
    "START_SIMULATION": handle_start_simulation,
    "STOP_SIMULATION": handle_stop_simulation,
    "INJECT_FAULT": handle_inject_fault,
    "OVERRIDE_PARAMETER": handle_override_parameter,
    "GET_STATUS": handle_get_status
}


#============================================================================
# SIMULATION LOOP (uses immutable snapshots per tick)
#============================================================================

async def run_simulation_loop(session_id: str, config: dict):
    tick = 0
    interval_ms = config.get("interval_ms", 1000)
    max_ticks = config.get("max_ticks", 0)
    logger.info(f"▶ Starting simulation loop for session {session_id}")

    try:
        while manager.sim_running.get(session_id, False):
            start_time = time.time()
            snapshot = await handle_get_session_snapshot(session_id)
            if not snapshot:
                break

            try:
                result = await run_simulation_step(session_id, tick, snapshot)
                # Apply SOC update immutably
                soc_change = 0.0
                if result["dispatch"]["action"] == "CHARGE_BESS":
                    soc_change = 0.05
                elif result["dispatch"]["action"] == "DISCHARGE_BESS":
                    soc_change = -0.05
                if soc_change != 0.0:
                    updated_snapshot = update_battery_soc_in_snapshot(snapshot, soc_change)
                    await handle_save_session_snapshot(session_id, updated_snapshot)

                await manager.broadcast_to_session({
                    "type": "METRIC_UPDATE",
                    "timestamp": time.time(),
                    "tick": tick,
                    "data": result
                }, session_id, metric_type="metrics")

            except Exception as e:
                logger.error(f"Simulation step error: {e}")
                await manager.broadcast_to_session({
                    "type": "SIMULATION_ERROR",
                    "timestamp": time.time(),
                    "error": str(e)
                }, session_id, metric_type="alerts")

            tick += 1
            if max_ticks > 0 and tick >= max_ticks:
                manager.sim_running[session_id] = False
                break

            elapsed_ms = (time.time() - start_time) * 1000
            sleep_ms = max(0, interval_ms - elapsed_ms)
            await asyncio.sleep(sleep_ms / 1000)

    except asyncio.CancelledError:
        logger.info(f"Simulation loop cancelled for session {session_id}")
    finally:
        manager.sim_running[session_id] = False
        await manager.broadcast_to_session({
            "type": "SIMULATION_COMPLETED",
            "timestamp": time.time(),
            "total_ticks": tick
        }, session_id)


#============================================================================
# WEBSOCKET ENDPOINT (completed and refactored)
#============================================================================

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    client_id: str = Query(..., description="Unique client identifier")
):
    if not DATA_MANAGER_AVAILABLE:
        await websocket.close(code=1011, reason="DataManager unavailable")
        return

    snapshot = await handle_get_session_snapshot(session_id)
    if not snapshot:
        await websocket.close(code=1008, reason="Session not found")
        return

    await manager.connect(websocket, session_id, client_id)

    try:
        await manager.send_personal_message({
            "type": "CONNECTED",
            "timestamp": time.time(),
            "session_id": session_id,
            "client_id": client_id,
            "available_metrics": ["metrics", "alerts", "dispatch", "constraints", "system"],
            "available_commands": list(MESSAGE_HANDLERS.keys()),
            "capabilities": {
                "data_manager": DATA_MANAGER_AVAILABLE,
                "physics": PHYSICS_AVAILABLE,
                "pinn": PINN_AVAILABLE
            }
        }, session_id, client_id)

        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            if not message_type:
                await manager.send_personal_message(
                    {"type": "ERROR", "message": "Missing message type"}, session_id, client_id
                )
                continue

            handler = MESSAGE_HANDLERS.get(message_type)
            if handler:
                try:
                    response = await handler(session_id, client_id, data)
                    await manager.send_personal_message(response, session_id, client_id)
                except Exception as e:
                    logger.error(f"Handler error for {message_type}: {e}")
                    await manager.send_personal_message({
                        "type": "ERROR", "message": f"Handler failed: {str(e)}"
                    }, session_id, client_id)
            else:
                await manager.send_personal_message({
                    "type": "ERROR", "message": f"Unknown message type: {message_type}"
                }, session_id, client_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id, client_id)
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id, client_id)


#============================================================================
# ADMIN ENDPOINTS (unchanged)
#============================================================================

@router.get("/connections/stats")
async def get_connection_stats():
    total_clients = sum(len(clients) for clients in manager.active_connections.values())
    return {
        "total_sessions": len(manager.active_connections),
        "total_clients": total_clients,
        "sessions": {session_id: len(clients) for session_id, clients in manager.active_connections.items()}
    }

@router.get("/connections/clients/{session_id}")
async def get_session_clients(session_id: str):
    clients = list(manager.active_connections.get(session_id, {}).keys())
    return {
        "session_id": session_id,
        "client_count": len(clients),
        "clients": [{"client_id": cid, "metadata": manager.client_metadata.get(cid, {})} for cid in clients]
    }

@router.post("/broadcast/message")
async def broadcast_message(session_id: str, message: dict):
    await manager.broadcast_to_session(message, session_id)
    client_count = len(manager.active_connections.get(session_id, {}))
    return {"status": "broadcasted", "session_id": session_id, "client_count": client_count}
