"""
D.E.C.O.D.E. Backend API Controller.
Exposes the Physics-Informed Digital Twin as a SaaS (Session-based).

Responsibilities:
1. Context Management: Loading specific IEEE Grids & Weather Profiles per user session.
2. Model Orchestration: Spawning training jobs for DeepONet architectures.
3. Real-Time Inference: Serving predictions with Physics Guardrails (Projection Layer).
"""

import logging
import uuid
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- Internal Service Imports ---
# Assumes the file structure defined in the plan
from backend.services.data_manager import data_manager
# We assume the user has applied the requested alterations to DeepONet
from ml_models.architectures.deeponet import DeepONet
# We assume the new projection layer is created
from physics_core.constraints.projection import project_to_feasible_manifold

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("API_Controller")

app = FastAPI(title="D.E.C.O.D.E. Industrial SaaS Engine", version="2.0-Alpha")

# --- In-Memory State Management (SaaS Session Store) ---
# In production, replace with Redis (State) + PostgreSQL (Metadata)
sessions: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Data Models (Input Validation) ---

class ContextConfig(BaseModel):
    """User input to define the simulation environment."""
    grid_topology: str = "ieee118"  # Options: ieee14, ieee118
    weather_profile: str = "solar_egypt"  # Options: solar_egypt, wind_north

class TrainRequest(BaseModel):
    """User input to trigger model training."""
    session_id: str
    epochs: int = 50
    learning_rate: float = 0.001

class PredictionRequest(BaseModel):
    """User input for real-time interaction."""
    session_id: str
    tick: int  # Hour of the day (0-23)
    load_scaling: float = 1.0  # Multiplier to simulate grid stress (e.g., 1.2 = +20% Load)

# --- Background Worker Functions ---

def _train_session_model(session_id: str, epochs: int, lr: float):
    """
    Background task that trains the DeepONet for the specific session context.
    Decoupled from the HTTP response to ensure non-blocking UI.
    """
    try:
        session = sessions[session_id]
        topology = session["topology"]
        weather = session["weather"]
        
        logger.info(f"⚙️ [Session {session_id}] Starting Training: {topology['name']} ({topology['n_buses']} Buses)")

        # 1. Initialize Architecture (Dynamic Resizing based on Grid)
        # The Trunk Net must match the number of buses in the selected IEEE grid
        n_buses = topology["n_buses"]
        model = DeepONet(
            input_dim=3,      # [Solar, Wind, Load]
            hidden_dim=64,
            output_dim=1,     # Voltage Magnitude per bus
            n_buses=n_buses   # <--- Dynamic Parameter (The alteration requested)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        # 2. Prepare Data (Synthetic Tensorization)
        # Converting the DataManager's list output to PyTorch tensors
        # Shape: [24 hours, 3 features]
        sensor_data = torch.tensor(np.stack([
            weather["ghi"], 
            weather["wind_speed"], 
            weather["temperature"] # Using Temp as proxy for Load profile base
        ], axis=1), dtype=torch.float32)
        
        # Domain: The Bus Indices [0, 1, ..., N-1]
        bus_ids = torch.arange(n_buses, dtype=torch.long).unsqueeze(0).repeat(24, 1) # [24, N]

        # 3. Training Loop (Simplified for SaaS Demo)
        model.train()
        final_loss = 0.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward Pass: Predict Voltage for ALL buses at ALL times
            # Input: [24, 3], Bus_IDs: [24, N]
            prediction = model(sensor_data, bus_ids) # Output: [24, N]
            
            # Target: Ideal Voltage is 1.0 p.u. (Flat start)
            target = torch.ones_like(prediction)
            
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"   Epoch {epoch}: Loss {final_loss:.5f}")

        # 4. Save State
        session["model"] = model
        session["status"] = "READY"
        session["metrics"] = {"final_loss": final_loss}
        
        logger.info(f"✅ [Session {session_id}] Training Complete. Loss: {final_loss:.4f}")

    except Exception as e:
        logger.error(f"❌ [Session {session_id}] Training Failed: {e}")
        session["status"] = "FAILED"
        session["error"] = str(e)

# --- API Endpoints ---

@app.get("/")
def health_check():
    return {
        "system": "D.E.C.O.D.E. SaaS Engine",
        "status": "Operational",
        "active_sessions": len(sessions)
    }

@app.post("/context/configure")
def configure_context(config: ContextConfig):
    """
    Step 1: User brings their context.
    API loads the 'Valuable Inputs' (Grid & Weather) into a session.
    """
    try:
        # Generate Session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Call Data Layer
        topology_data = data_manager.get_topology(config.grid_topology)
        weather_data = data_manager.get_weather_profile(config.weather_profile)
        
        # Store State
        sessions[session_id] = {
            "config": config.dict(),
            "topology": topology_data,
            "weather": weather_data,
            "status": "CONFIGURED",
            "model": None
        }
        
        logger.info(f"📝 Session {session_id} configured for {config.grid_topology}")
        
        return {
            "session_id": session_id,
            "message": "Context Loaded Successfully",
            "grid_summary": {
                "name": topology_data["name"],
                "buses": topology_data["n_buses"],
                "lines": topology_data["n_lines"]
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/model/train")
def trigger_training(payload: TrainRequest, background_tasks: BackgroundTasks):
    """
    Step 2: User requests the 'Digital Twin' build.
    Triggers asynchronous training for the specific grid topology.
    """
    if payload.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session ID not found")
    
    current_status = sessions[payload.session_id]["status"]
    if current_status != "CONFIGURED":
        raise HTTPException(status_code=400, detail=f"Cannot train. Current status: {current_status}")

    # Update Status
    sessions[payload.session_id]["status"] = "TRAINING"
    
    # Dispatch Background Task
    background_tasks.add_task(
        _train_session_model, 
        payload.session_id, 
        payload.epochs, 
        payload.learning_rate
    )
    
    return {
        "session_id": payload.session_id,
        "status": "Training Started",
        "estimated_time": f"{payload.epochs * 0.1:.1f} seconds"
    }

@app.get("/model/status/{session_id}")
def check_status(session_id: str):
    """Polling endpoint for the CLI/Frontend to check training progress."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": sessions[session_id]["status"],
        "metrics": sessions[session_id].get("metrics", {})
    }

@app.post("/simulation/predict")
def run_simulation_step(payload: PredictionRequest):
    """
    Step 3: User interacts with the trained Digital Twin.
    Runs inference, checks Physics Constraints, and decides Control Action.
    """
    # 1. Validation
    if payload.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[payload.session_id]
    if session["status"] != "READY" or session["model"] is None:
        raise HTTPException(status_code=400, detail="Model is not ready. Train it first.")

    # 2. Prepare Inputs
    tick = payload.tick % 24
    weather = session["weather"]
    
    # Input Vector: [Solar, Wind, Load_Request]
    # We apply the user's load_scaling to the temperature/base load
    current_load = weather["temperature"][tick] * payload.load_scaling
    
    input_tensor = torch.tensor([[
        weather["ghi"][tick],
        weather["wind_speed"][tick],
        current_load
    ]], dtype=torch.float32)
    
    # Bus Vector: Predict for ALL buses
    n_buses = session["topology"]["n_buses"]
    bus_ids = torch.arange(n_buses, dtype=torch.long).unsqueeze(0) # [1, N]

    # 3. Model Inference (DeepONet)
    model = session["model"]
    model.eval()
    with torch.no_grad():
        # Raw Prediction from Neural Network
        raw_voltage = model(input_tensor, bus_ids) # [1, N]

    # 4. Physics Guardrail (The Safety Valve)
    # Project the raw neural prediction onto the feasible voltage manifold (0.9 - 1.1 p.u.)
    topology_adj = session["topology"]["adj_matrix"]
    
    # Call the projection layer logic
    valid_voltage, violation_info = project_to_feasible_manifold(
        raw_voltage, 
        topology_adj,
        limits=(0.9, 1.1)
    )
    
    # 5. Control Logic (The Energy Router)
    # If Net Load is negative (Generation > Consumption), Charge Battery
    net_generation = weather["ghi"][tick] - (current_load * 10) # Simple coefficient
    
    action = "IDLE"
    if net_generation > 50:
        action = "CHARGE_BATTERY"
    elif net_generation < -50:
        action = "DISCHARGE_BATTERY"
    
    # 6. Response Construction
    return {
        "tick": tick,
        "inputs": {
            "solar_w_m2": round(weather["ghi"][tick], 2),
            "wind_m_s": round(weather["wind_speed"][tick], 2),
            "grid_load_pu": round(payload.load_scaling, 2)
        },
        "grid_state": {
            "avg_voltage_pu": float(valid_voltage.mean()),
            "min_voltage_pu": float(valid_voltage.min()),
            "stability_index": float(1.0 - violation_info["violation_magnitude"])
        },
        "safety_system": {
            "physics_violation_detected": violation_info["was_violated"],
            "correction_applied": violation_info["correction_magnitude"]
        },
        "control_decision": action
    }
