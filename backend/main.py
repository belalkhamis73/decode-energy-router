"""
D.E.C.O.D.E. Backend Master Controller (v3.1 - Harmonized)
Strictly integrates all Physics, ML, and Service layers.

Fixes applied:
1. Removed Legacy 'Split Brain' Logic (local sessions dict).
2. Routed ALL simulation traffic to the robust 'simulation.py' handler.
3. Centralized State Management via DataManager.
"""

import sys
import os
import uuid
import logging
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

# --- PATH SETUP ---
sys.path.append(os.getcwd())

# --- IMPORTS ---
from backend.services.data_manager import data_manager
from backend.app.core.pinn_engine import pinn_engine
from backend.app.api.routes import simulation  # The Real Logic

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MasterController")

# --- APP SETUP ---
app = FastAPI(title="D.E.C.O.D.E. Integrated Energy SaaS", version="3.1.0")

# --- CRITICAL FIX: ROUTING ---
# We mount the simulation router at the root "/simulation" path.
# This ensures the CLI (which hits /simulation/predict) talks to the NEW logic,
# not the old legacy function.
app.include_router(simulation.router, prefix="/simulation", tags=["simulation"])

# --- CONFIGURATION DTO ---
class ConfigReq(BaseModel):
    grid_topology: str
    weather_profile: str
    scenario: str = "normal"

# --- SYSTEM ENDPOINTS ---

@app.get("/")
def health_check():
    """Verifies the system is online and layers are connected."""
    return {
        "status": "active",
        "mode": "integrated",
        "router": "simulation.py (Mounted at /simulation)",
        "state_store": "DataManager (In-Memory)"
    }

@app.post("/context/configure")
def configure_session(c: ConfigReq):
    """
    Initializes a Digital Twin Session.
    Delegates state creation to DataManager to ensure consistency.
    """
    sid = str(uuid.uuid4())[:8]
    
    # 1. Create Persistent Session (State)
    # This stores the specific IEEE topology, Weather, and Battery State
    session = data_manager.create_session(sid, c.grid_topology, c.weather_profile)
    
    # 2. Register Model (Intelligence)
    # This ensures the PINN Engine knows which grid size (14 vs 118) to use
    pinn_engine.register_model(sid, n_buses=session['topology']['n_buses'])
    
    logger.info(f"✅ Session {sid} Configured. Topology: {session['topology']['name']}")
    
    return {
        "session_id": sid, 
        "grid_summary": session['topology'], 
        "message": "Digital Twin Initialized."
    }

@app.post("/model/train")
def train_model(p: dict, bt: BackgroundTasks):
    """
    Trigger background training for the specific session model.
    """
    sid = p.get('session_id')
    if not sid:
        return {"error": "session_id required"}
        
    def _orchestrate_training():
        # In a real system, this calls train_pinn.py
        logger.info(f"🏋️‍♂️ Training Model for Session {sid}...")
        # Simulate training delay
        import time
        time.sleep(1.5) 
        
        # Mark session as ready in DataManager/PINN
        # (For this demo, we assume the pre-loaded weights or mock is ready)
        logger.info(f"✅ Training complete for {sid}")
        
    bt.add_task(_orchestrate_training)
    return {"status": "Training Job Dispatched", "session_id": sid}

# --- SERVER LAUNCHER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
