"""
WebSocket Route Handler - Real-time Bidirectional Communication
Provides live streaming of:
1. Model predictions (6 parallel models)
2. Dispatch decisions and routing logic
3. Constraint violations and alerts
4. Asset health metrics
5. Financial KPIs

Supports:
- Multiple concurrent client connections
- Selective metric subscriptions
- Control commands (start/stop sim, inject faults, override params)
- Event-driven state broadcasts
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Dict, Set, Optional, Any, List
import asyncio
import json
import logging
import time
from datetime import datetime
from collections import defaultdict
import torch

# Import from simulation route
from backend.app.schemas.fault_payload import FaultPayload
from backend.services.data_manager import data_manager

# Import physics models
from physics_core.equations.energy_router import EnergyRouter, SourceState
from physics_core.equations.grid_estimator import GridEstimator
from physics_core.constraints.projection import project_to_feasible_manifold

# Import ML engine
from backend.app.core.pinn_engine import pinn_engine

router = APIRouter()
logger = logging.getLogger("WebSocketRoute")

# Initialize physics engines
energy_router = EnergyRouter()
grid_estimator = GridEstimator()


# ============================================================================
# CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections with subscription support."""
    
    def __init__(self):
        # Active connections: {session_id: {client_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = defaultdict(dict)
        
        # Client subscriptions: {client_id: Set[metric_type]}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Client metadata
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Simulation state per session
        self.sim_running: Dict[str, bool] = {}
        self.sim_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(
        self, 
        websocket: WebSocket, 
        session_id: str, 
        client_id: str
    ):
        """Accept and register new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id][client_id] = websocket
        self.client_metadata[client_id] = {
            "session_id": session_id,
            "connected_at": time.time(),
            "message_count": 0
        }
        logger.info(f"✅ Client {client_id} connected to session {session_id}")
        
    def disconnect(self, session_id: str, client_id: str):
        """Remove client connection."""
        if client_id in self.active_connections.get(session_id, {}):
            del self.active_connections[session_id][client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
        logger.info(f"❌ Client {client_id} disconnected from session {session_id}")
        
    def subscribe(self, client_id: str, metrics: List[str]):
        """Add metric subscriptions for client."""
        self.subscriptions[client_id].update(metrics)
        logger.debug(f"Client {client_id} subscribed to: {metrics}")
        
    def unsubscribe(self, client_id: str, metrics: List[str]):
        """Remove metric subscriptions."""
        self.subscriptions[client_id].difference_update(metrics)
        logger.debug(f"Client {client_id} unsubscribed from: {metrics}")
        
    async def send_personal_message(
        self, 
        message: dict, 
        session_id: str, 
        client_id: str
    ):
        """Send message to specific client."""
        websocket = self.active_connections.get(session_id, {}).get(client_id)
        if websocket:
            try:
                await websocket.send_json(message)
                self.client_metadata[client_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                
    async def broadcast_to_session(
        self, 
        message: dict, 
        session_id: str,
        metric_type: Optional[str] = None
    ):
        """Broadcast message to all clients in session (with subscription filter)."""
        if session_id not in self.active_connections:
            return
            
        tasks = []
        for client_id, websocket in self.active_connections[session_id].items():
            # Check subscription filter
            if metric_type and metric_type not in self.subscriptions.get(client_id, set()):
                continue
                
            try:
                tasks.append(websocket.send_json(message))
                self.client_metadata[client_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Broadcast error to {client_id}: {e}")
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def broadcast_to_all(self, message: dict):
        """Broadcast to all connected clients."""
        for session_id in self.active_connections.keys():
            await self.broadcast_to_session(message, session_id)
            
    def get_session_clients(self, session_id: str) -> List[str]:
        """Get list of client IDs for a session."""
        return list(self.active_connections.get(session_id, {}).keys())
        
    def get_connection_stats(self) -> dict:
        """Get connection statistics."""
        total_clients = sum(len(clients) for clients in self.active_connections.values())
        return {
            "total_sessions": len(self.active_connections),
            "total_clients": total_clients,
            "sessions": {
                session_id: len(clients)
                for session_id, clients in self.active_connections.items()
            }
        }


manager = ConnectionManager()


# ============================================================================
# MESSAGE HANDLERS
# ============================================================================

async def handle_subscribe(client_id: str, payload: dict):
    """Handle subscription request."""
    metrics = payload.get("metrics", [])
    if metrics:
        manager.subscribe(client_id, metrics)
        return {
            "type": "SUBSCRIPTION_CONFIRMED",
            "metrics": list(manager.subscriptions[client_id])
        }
    return {"type": "ERROR", "message": "No metrics specified"}


async def handle_unsubscribe(client_id: str, payload: dict):
    """Handle unsubscription request."""
    metrics = payload.get("metrics", [])
    if metrics:
        manager.unsubscribe(client_id, metrics)
        return {
            "type": "UNSUBSCRIPTION_CONFIRMED",
            "metrics": list(manager.subscriptions[client_id])
        }
    return {"type": "ERROR", "message": "No metrics specified"}


async def handle_start_simulation(session_id: str, client_id: str, payload: dict):
    """Start continuous simulation loop."""
    if manager.sim_running.get(session_id):
        return {"type": "ERROR", "message": "Simulation already running"}
        
    manager.sim_running[session_id] = True
    
    # Start simulation task
    task = asyncio.create_task(
        run_simulation_loop(session_id, payload)
    )
    manager.sim_tasks[session_id] = task
    
    return {
        "type": "SIMULATION_STARTED",
        "session_id": session_id,
        "config": payload
    }


async def handle_stop_simulation(session_id: str, client_id: str, payload: dict):
    """Stop continuous simulation."""
    if not manager.sim_running.get(session_id):
        return {"type": "ERROR", "message": "No simulation running"}
        
    manager.sim_running[session_id] = False
    
    # Cancel task
    if session_id in manager.sim_tasks:
        manager.sim_tasks[session_id].cancel()
        del manager.sim_tasks[session_id]
        
    return {
        "type": "SIMULATION_STOPPED",
        "session_id": session_id
    }


async def handle_inject_fault(session_id: str, client_id: str, payload: dict):
    """Inject fault into simulation."""
    try:
        fault = FaultPayload(**payload.get("fault", {}))
        session = data_manager.get_session(session_id)
        if not session:
            return {"type": "ERROR", "message": "Session not found"}
            
        session['active_fault'] = fault.dict()
        
        # Broadcast fault event
        await manager.broadcast_to_session(
            {
                "type": "FAULT_INJECTED",
                "timestamp": time.time(),
                "fault": fault.dict()
            },
            session_id,
            metric_type="alerts"
        )
        
        return {
            "type": "FAULT_INJECTION_CONFIRMED",
            "fault": fault.dict()
        }
    except Exception as e:
        return {"type": "ERROR", "message": f"Fault injection failed: {str(e)}"}


async def handle_override_parameter(session_id: str, client_id: str, payload: dict):
    """Override simulation parameter."""
    try:
        path = payload.get("parameter_path")
        value = payload.get("value")
        
        if not path or value is None:
            return {"type": "ERROR", "message": "Missing path or value"}
            
        session = data_manager.get_session(session_id)
        if not session:
            return {"type": "ERROR", "message": "Session not found"}
            
        # Apply override
        keys = path.split('.')
        target = session
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
        
        # Broadcast override event
        await manager.broadcast_to_session(
            {
                "type": "PARAMETER_OVERRIDDEN",
                "timestamp": time.time(),
                "parameter_path": path,
                "value": value
            },
            session_id,
            metric_type="system"
        )
        
        return {
            "type": "OVERRIDE_CONFIRMED",
            "parameter_path": path,
            "value": value
        }
    except Exception as e:
        return {"type": "ERROR", "message": f"Override failed: {str(e)}"}


async def handle_get_status(session_id: str, client_id: str, payload: dict):
    """Get current simulation status."""
    session = data_manager.get_session(session_id)
    if not session:
        return {"type": "ERROR", "message": "Session not found"}
        
    return {
        "type": "STATUS_RESPONSE",
        "timestamp": time.time(),
        "simulation_running": manager.sim_running.get(session_id, False),
        "session_state": {
            "battery_soc": session.get("battery_soc", 0.5),
            "battery_temp_c": session.get("battery_temp_c", 25.0),
            "battery_soh": session.get("battery_soh", 1.0),
            "diesel_status": session.get("diesel_status", "OFF")
        },
        "active_fault": session.get("active_fault"),
        "connected_clients": len(manager.active_connections.get(session_id, {}))
    }


# Message handler registry
MESSAGE_HANDLERS = {
    "SUBSCRIBE": handle_subscribe,
    "UNSUBSCRIBE": handle_unsubscribe,
    "START_SIMULATION": handle_start_simulation,
    "STOP_SIMULATION": handle_stop_simulation,
    "INJECT_FAULT": handle_inject_fault,
    "OVERRIDE_PARAMETER": handle_override_parameter,
    "GET_STATUS": handle_get_status
}


# ============================================================================
# SIMULATION LOOP
# ============================================================================

async def run_simulation_loop(session_id: str, config: dict):
    """Continuous simulation loop broadcasting updates."""
    tick = 0
    interval_ms = config.get("interval_ms", 1000)
    max_ticks = config.get("max_ticks", 0)  # 0 = infinite
    
    logger.info(f"🔄 Starting simulation loop for session {session_id}")
    
    try:
        while manager.sim_running.get(session_id, False):
            start_time = time.time()
            
            # Get session state
            session = data_manager.get_session(session_id)
            if not session:
                break
                
            # Run simulation step (simplified version)
            try:
                result = await run_simulation_step(session_id, tick, session)
                
                # Broadcast metrics
                await manager.broadcast_to_session(
                    {
                        "type": "METRIC_UPDATE",
                        "timestamp": time.time(),
                        "tick": tick,
                        "data": result
                    },
                    session_id,
                    metric_type="metrics"
                )
                
            except Exception as e:
                logger.error(f"Simulation step error: {e}")
                await manager.broadcast_to_session(
                    {
                        "type": "SIMULATION_ERROR",
                        "timestamp": time.time(),
                        "error": str(e)
                    },
                    session_id,
                    metric_type="alerts"
                )
                
            # Increment tick
            tick += 1
            if max_ticks > 0 and tick >= max_ticks:
                manager.sim_running[session_id] = False
                break
                
            # Sleep to maintain interval
            elapsed_ms = (time.time() - start_time) * 1000
            sleep_ms = max(0, interval_ms - elapsed_ms)
            await asyncio.sleep(sleep_ms / 1000)
            
    except asyncio.CancelledError:
        logger.info(f"Simulation loop cancelled for session {session_id}")
    finally:
        manager.sim_running[session_id] = False
        await manager.broadcast_to_session(
            {
                "type": "SIMULATION_COMPLETED",
                "timestamp": time.time(),
                "total_ticks": tick
            },
            session_id
        )


async def run_simulation_step(session_id: str, tick: int, session: dict) -> dict:
    """Execute single simulation step (from simulation.py logic)."""
    weather = session['weather']
    tick_idx = tick % 24
    
    # Extract environment
    ghi = weather['ghi'][tick_idx]
    temp = weather['temperature'][tick_idx]
    wind = weather['wind_speed'][tick_idx]
    cloud_cover = weather.get('cloud_cover', [0.3] * 24)[tick_idx]
    
    # Calculate load
    base_load_kw = temp * 10.0
    
    # Apply active fault if any
    active_fault = session.get('active_fault')
    if active_fault and active_fault.get('is_active'):
        base_load_kw *= (1.0 + active_fault.get('magnitude_pu', 0.0))
    
    # Simplified model predictions (inline for speed)
    solar_eff = max(0.0, 0.18 - (temp - 25) * 0.005) * (1 - cloud_cover * 0.7)
    solar_kw = ghi * solar_eff * 100.0
    
    wind_kw = 0.0
    if 3.5 < wind < 25.0:
        wind_kw = 0.5 * 1.225 * (wind ** 3) * 0.4 if wind < 12.0 else 50.0
    
    # Energy routing
    source_state = SourceState(
        solar_kw=solar_kw,
        wind_kw=wind_kw,
        battery_soc=session.get('battery_soc', 0.5),
        v2g_available_kw=weather.get('v2g_availability_kw', [0] * 24)[tick_idx],
        load_demand_kw=base_load_kw,
        diesel_status=session.get('diesel_status', 'OFF')
    )
    
    dispatch = energy_router.compute_dispatch(source_state, 1.0)
    
    # Update battery SOC
    soc_change = 0.0
    if dispatch['action'] == 'CHARGE_BESS':
        soc_change = 0.05
    elif dispatch['action'] == 'DISCHARGE_BESS':
        soc_change = -0.05
    
    new_soc = max(0.0, min(1.0, session.get('battery_soc', 0.5) + soc_change))
    session['battery_soc'] = new_soc
    
    # Check constraints
    constraints = []
    if new_soc < 0.2:
        constraints.append("BATTERY_SOC_CRITICAL")
    elif new_soc > 0.9:
        constraints.append("BATTERY_SOC_HIGH")
    
    # Return metrics
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
            "battery_temp_c": session.get('battery_temp_c', 25.0)
        },
        "constraints": constraints
    }


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    client_id: str = Query(..., description="Unique client identifier")
):
    """
    Main WebSocket endpoint for real-time simulation communication.
    
    CLIENT → SERVER Messages:
    - SUBSCRIBE: {"type": "SUBSCRIBE", "metrics": ["metrics", "alerts"]}
    - UNSUBSCRIBE: {"type": "UNSUBSCRIBE", "metrics": ["metrics"]}
    - START_SIMULATION: {"type": "START_SIMULATION", "interval_ms": 1000}
    - STOP_SIMULATION: {"type": "STOP_SIMULATION"}
    - INJECT_FAULT: {"type": "INJECT_FAULT", "fault": {...}}
    - OVERRIDE_PARAMETER: {"type": "OVERRIDE_PARAMETER", "parameter_path": "...", "value": ...}
    - GET_STATUS: {"type": "GET_STATUS"}
    
    SERVER → CLIENT Messages:
    - METRIC_UPDATE: Real-time simulation metrics
    - FAULT_INJECTED: Fault event notification
    - PARAMETER_OVERRIDDEN: Parameter change notification
    - SIMULATION_ERROR: Error during simulation
    - STATUS_RESPONSE: Current status response
    """
    
    # Verify session exists
    session = data_manager.get_session(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    await manager.connect(websocket, session_id, client_id)
    
    try:
        # Send welcome message
        await manager.send_personal_message(
            {
                "type": "CONNECTED",
                "timestamp": time.time(),
                "session_id": session_id,
                "client_id": client_id,
                "available_metrics": [
                    "metrics",
                    "alerts",
                    "dispatch",
                    "constraints",
                    "system"
                ],
                "available_commands": list(MESSAGE_HANDLERS.keys())
            },
            session_id,
            client_id
        )
        
        # Message loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            if not message_type:
                await manager.send_personal_message(
                    {"type": "ERROR", "message": "Missing message type"},
                    session_id,
                    client_id
                )
                continue
            
            # Handle message
            handler = MESSAGE_HANDLERS.get(message_type)
            if handler:
                try:
                    response = await handler(session_id, client_id, data)
                    await manager.send_personal_message(
                        response,
                        session_id,
                        client_id
                    )
                except Exception as e:
                    logger.error(f"Handler error for {message_type}: {e}")
                    await manager.send_personal_message(
                        {
                     
