"""
Simulation Routes Module.
Handles requests for High-Fidelity Grid Simulations and Contingency Analysis.
Bridges the gap between the REST API (FastAPI) and the Core Physics Engine (Pandapower).
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from typing import Dict, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import Data Contracts (DTOs)
from app.schemas.fault_payload import FaultScenario, SimulationResult, GridState

# Import Core Engine (Singleton or Service)
# In a full microservices architecture, this might call a separate Worker Service via Redis/Celery.
# For this Mono-Repo MVP, we import the engine directly but run it off the main loop.
from app.core.config import settings

# Setup Logger
logger = logging.getLogger("Simulation_Route")

router = APIRouter()

# ThreadPool for CPU-bound physics calculations (Pandapower)
# Prevents blocking the AsyncIO event loop
executor = ThreadPoolExecutor(max_workers=3)

def run_physics_engine(scenario: FaultScenario) -> Dict[str, Any]:
    """
    Wrapper for the heavy physics simulation.
    This would typically invoke Pandapower or PyPSA.
    """
    logger.info(f"Starting Physics Simulation: {scenario.fault_type} on {scenario.target_component}")
    
    # --- MOCKING THE PHYSICS ENGINE FOR API LAYER ---
    # In integration phase, replace this with:
    # net = pp.create_empty_network()
    # pp.runpp(net)
    
    # Simulating computation time
    import time
    time.sleep(0.5) 
    
    # Return mock results compliant with Physics Laws (e.g., Voltage drops)
    # Fail Fast logic: If fault is critical, show collapse
    is_stable = True
    voltage_pu = 0.98
    
    if scenario.magnitude > 0.8: # Massive fault
        is_stable = False
        voltage_pu = 0.65 # Voltage collapse
        
    return {
        "is_stable": is_stable,
        "voltage_profile_pu": [voltage_pu] * 14, # 14-Bus example
        "frequency_hz": 59.2 if not is_stable else 60.0,
        "converged": True
    }

@router.post("/run", response_model=SimulationResult, status_code=status.HTTP_200_OK)
async def trigger_simulation(
    scenario: FaultScenario, 
    background_tasks: BackgroundTasks
):
    """
    Trigger a synchronous High-Fidelity Simulation.
    Used for 'What-If' analysis in the Control Room.
    
    Process:
    1. Validate Fault Scenario (Pydantic).
    2. Offload heavy calculation to ThreadPool (don't block API).
    3. Return physical grid state.
    """
    try:
        # 1. Fail Fast: Validate basic physics constraints before running
        if scenario.duration_ms < 0:
            raise HTTPException(status_code=400, detail="Time cannot be negative (Causality Violation).")

        # 2. Execution Strategy:
        # Run CPU-bound physics in a separate thread to keep API responsive (AsyncIO non-blocking)
        loop = asyncio.get_event_loop()
        result_data = await loop.run_in_executor(executor, run_physics_engine, scenario)
        
        # 3. Construct Response
        return SimulationResult(
            timestamp=scenario.timestamp,
            grid_state=GridState(
                voltages=result_data["voltage_profile_pu"],
                frequency=result_data["frequency_hz"],
                stability_margin=0.15 if result_data["is_stable"] else -0.5
            ),
            physics_violation_detected=not result_data["is_stable"]
        )

    except Exception as e:
        logger.error(f"Simulation Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Physics Engine Error")

@router.post("/stress-test/async", status_code=status.HTTP_202_ACCEPTED)
async def trigger_stress_test_async(
    scenario: FaultScenario, 
    background_tasks: BackgroundTasks
):
    """
    Fire-and-Forget endpoint for massive contingency screening.
    Returns a Task ID immediately; processing happens in background.
    Ideal for 'Track A' demo where UI shouldn't freeze.
    """
    task_id = f"sim_{hash(scenario.timestamp)}"
    
    # Add to background tasks (Simple Queue)
    # In Production, this line is replaced by: celery_app.send_task("simulate_fault", args=[...])
    background_tasks.add_task(run_physics_engine, scenario)
    
    return {"task_id": task_id, "status": "queued", "message": "Digital Twin is crunching numbers..."}
  
