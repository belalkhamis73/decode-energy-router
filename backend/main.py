"""
D.E.C.O.D.E. Backend Master Controller (v4.0 - Streaming Architecture)
Future-Ready Real-Time Multi-Physics Streaming with WebSockets, SSE, and Async Orchestration.

Migration Path:
- v3.1: Synchronous REST API with in-memory state
- v4.0: Async streaming architecture with Redis pub/sub (THIS VERSION)
- Future: Kubernetes-native with TimescaleDB persistence

Compatibility Layer:
- Preserves v3.1 synchronous endpoints for backward compatibility
- New streaming endpoints operate in parallel
- Graceful degradation when Redis/advanced services unavailable
"""

import os
import sys
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- PATH SETUP ---
sys.path.append(os.getcwd())

# --- CORE v3.1 IMPORTS (Always Available) ---
from backend.services.data_manager import data_manager
from backend.app.core.pinn_engine import pinn_engine
from backend.app.api.routes import simulation

# --- v4.0 STREAMING IMPORTS (Graceful Fallback) ---
STREAMING_ENABLED = False
try:
    import redis.asyncio as redis
    from backend.services.model_orchestrator import ModelOrchestrator
    from backend.services.streaming_data_manager import streaming_data_manager
    from backend.services.control_plane import control_plane
    STREAMING_ENABLED = True
except ImportError as e:
    logging.warning(f"⚠️ Streaming features disabled: {e}")
    logging.info("Running in v3.1 compatibility mode")

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MasterController")

# --- APP SETUP ---
app = FastAPI(
    title="D.E.C.O.D.E. Integrated Energy SaaS",
    description="Multi-Physics Digital Twin for Industrial Microgrids",
    version="4.0.0"
)

# --- REDIS CONNECTION POOL (Conditional) ---
redis_client = None
orchestrator = None

if STREAMING_ENABLED:
    try:
        redis_pool = redis.ConnectionPool.from_url(
            "redis://localhost:6379",
            decode_responses=True,
            max_connections=20
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        orchestrator = ModelOrchestrator()
        logger.info("✅ Streaming Architecture Initialized")
    except Exception as e:
        logger.error(f"❌ Redis Connection Failed: {e}")
        STREAMING_ENABLED = False

# --- v3.1 BACKWARD COMPATIBILITY ROUTES ---
app.include_router(simulation.router, prefix="/simulation", tags=["simulation"])

# --- SHARED SCHEMAS ---
class ConfigReq(BaseModel):
    grid_topology: str
    weather_profile: str
    scenario: str = "normal"

class ScenarioInjection(BaseModel):
    scenario_type: str  # "solar_eclipse", "cyber_attack", "generator_trip"
    magnitude: float
    duration_sec: int

# --- ACTIVE SESSION TRACKING ---
active_loops: Dict[str, asyncio.Task] = {}

# ==================== HEALTH & DISCOVERY ====================

@app.get("/")
def health_check():
    """System health and capability discovery."""
    return {
        "status": "active",
        "version": "4.0.0",
        "mode": "streaming" if STREAMING_ENABLED else "legacy",
        "capabilities": {
            "synchronous_api": True,
            "websocket_streaming": STREAMING_ENABLED,
            "sse_monitoring": STREAMING_ENABLED,
            "redis_pubsub": STREAMING_ENABLED,
            "scenario_injection": STREAMING_ENABLED
        },
        "endpoints": {
            "legacy": {
                "configure": "/context/configure",
                "predict": "/simulation/predict"
            },
            "streaming": {
                "websocket": "/ws/stream/{session_id}",
                "sse": "/stream/metrics/{session_id}",
                "control": "/control/scenario/{session_id}",
                "metrics": "/metrics/live/{session_id}",
                "export": "/export/timeseries/{session_id}"
            } if STREAMING_ENABLED else None
        }
    }

# ==================== v3.1 LEGACY ENDPOINTS ====================

@app.post("/context/configure")
async def configure_session(c: ConfigReq, background_tasks: BackgroundTasks):
    """
    Initialize Digital Twin Session.
    - v3.1 mode: Simple state initialization
    - v4.0 mode: Starts continuous 1Hz simulation loop
    """
    sid = str(uuid.uuid4())[:8]
    
    # Create session in legacy data manager
    session = data_manager.create_session(sid, c.grid_topology, c.weather_profile)
    pinn_engine.register_model(sid, n_buses=session['topology']['n_buses'])
    
    logger.info(f"✅ Session {sid} Configured: {session['topology']['name']}")
    
    response = {
        "session_id": sid,
        "grid_summary": session['topology'],
        "message": "Digital Twin Initialized"
    }
    
    # If streaming enabled, start background loop
    if STREAMING_ENABLED and orchestrator:
        try:
            # Initialize streaming infrastructure
            await streaming_data_manager.init_session(sid, c.dict())
            await orchestrator.register_session_models(sid, c.dict())
            
            # Launch 1Hz simulation loop
            loop_task = asyncio.create_task(simulation_loop(sid))
            active_loops[sid] = loop_task
            
            response["streaming"] = {
                "websocket": f"/ws/stream/{sid}",
                "sse": f"/stream/metrics/{sid}",
                "control": f"/control/scenario/{sid}"
            }
            logger.info(f"🚀 Streaming Loop Started for {sid}")
        except Exception as e:
            logger.error(f"⚠️ Streaming initialization failed: {e}")
    
    return response

@app.post("/model/train")
def train_model(p: dict, bt: BackgroundTasks):
    """Trigger background training (v3.1 compatible)."""
    sid = p.get('session_id')
    if not sid:
        return {"error": "session_id required"}
    
    def _orchestrate_training():
        logger.info(f"🏋️‍♂️ Training Model for Session {sid}...")
        import time
        time.sleep(1.5)
        logger.info(f"✅ Training complete for {sid}")
    
    bt.add_task(_orchestrate_training)
    return {"status": "Training Job Dispatched", "session_id": sid}

# ==================== v4.0 STREAMING FEATURES ====================

async def simulation_loop(session_id: str):
    """
    Continuous 1Hz Simulation Loop.
    Executes parallel multi-physics inference and publishes to Redis.
    """
    logger.info(f"🔄 Starting 1Hz Loop: {session_id}")
    try:
        while True:
            # Fetch live snapshot
            current_state = await streaming_data_manager.get_live_snapshot(session_id)
            
            # Get control overrides
            overrides = await control_plane.get_active_overrides(session_id)
            
            # Multi-model parallel inference
            predictions = await orchestrator.run_parallel_inference(
                session_id, current_state, overrides
            )
            
            # Publish to Redis for WebSocket/SSE consumers
            await redis_client.publish(
                f"metrics:{session_id}",
                predictions.json()
            )
            
            # Maintain 1Hz cadence
            await asyncio.sleep(1.0)
            
    except asyncio.CancelledError:
        logger.info(f"🛑 Loop Stopped: {session_id}")
    except Exception as e:
        logger.error(f"❌ Loop Error [{session_id}]: {e}")

@app.websocket("/ws/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Real-time bidirectional telemetry and control."""
    if not STREAMING_ENABLED:
        await websocket.close(code=1003, reason="Streaming not available")
        return
    
    await websocket.accept()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"metrics:{session_id}")
    
    logger.info(f"🔌 WebSocket Connected: {session_id}")
    
    try:
        async def receive_commands():
            """Handle incoming control commands."""
            while True:
                data = await websocket.receive_text()
                logger.info(f"📨 Command Received: {data}")
                # Process control commands here
                
        async def send_metrics():
            """Stream metrics to client."""
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await websocket.send_text(message["data"])
        
        await asyncio.gather(receive_commands(), send_metrics())
        
    except WebSocketDisconnect:
        await pubsub.unsubscribe(f"metrics:{session_id}")
        logger.info(f"🔌 WebSocket Disconnected: {session_id}")

@app.get("/stream/metrics/{session_id}")
async def sse_metrics(session_id: str):
    """Server-Sent Events for unidirectional monitoring."""
    if not STREAMING_ENABLED:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"metrics:{session_id}")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield f"data: {message['data']}\n\n"
        finally:
            await pubsub.unsubscribe(f"metrics:{session_id}")
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/control/scenario/{session_id}")
async def inject_scenario(session_id: str, scenario: ScenarioInjection):
    """Runtime parameter/fault injection."""
    if not STREAMING_ENABLED:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    success = await control_plane.apply_scenario(session_id, scenario)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid scenario parameters")
    
    return {
        "status": "Scenario Injected",
        "session_id": session_id,
        "scenario": scenario.scenario_type,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics/live/{session_id}")
async def get_live_metrics(session_id: str):
    """Expose all internal model states."""
    if not STREAMING_ENABLED:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    return await orchestrator.get_full_stack_metrics(session_id)

@app.get("/export/timeseries/{session_id}")
async def export_timeseries(
    session_id: str,
    start_time: str,
    end_time: str
):
    """Extract historical data for post-event analysis."""
    if not STREAMING_ENABLED:
        raise HTTPException(status_code=503, detail="Streaming not available")
    
    data = await streaming_data_manager.get_historical_range(
        session_id, start_time, end_time
    )
    return {"session_id": session_id, "data": data, "count": len(data)}

@app.delete("/context/{session_id}")
async def terminate_session(session_id: str):
    """Gracefully shutdown session and cleanup resources."""
    # Stop streaming loop if running
    if session_id in active_loops:
        active_loops[session_id].cancel()
        del active_loops[session_id]
        logger.info(f"🛑 Terminated Loop: {session_id}")
    
    # Cleanup data manager
    if hasattr(data_manager, 'remove_session'):
        data_manager.remove_session(session_id)
    
    return {"status": "Session Terminated", "session_id": session_id}

# ==================== LIFECYCLE HOOKS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("🚀 D.E.C.O.D.E. v4.0 Starting...")
    
    if STREAMING_ENABLED and redis_client:
        try:
            await redis_client.ping()
            logger.info("✅ Redis Connection Verified")
        except Exception as e:
            logger.error(f"❌ Redis Ping Failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting Down...")
    
    # Cancel all active loops
    for sid, task in active_loops.items():
        task.cancel()
        logger.info(f"Cancelled loop: {sid}")
    
    # Close Redis connections
    if redis_client:
        await redis_client.close()

# ==================== SERVER LAUNCHER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
                                                     )
