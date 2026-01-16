from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from app.core.pinn_engine import PINNEngine

# --- Data Contracts (DTOs) ---
class SolarStateInput(BaseModel):
    """
    Validates the telemetry window. 
    Strictly enforces 24-hour sequence length to match LSTM training.
    """
    timestamp: str
    telemetry_window: List[List[float]] = Field(..., min_length=24, max_length=24)

class GridPrediction(BaseModel):
    ghi_forecast: float
    confidence_score: float = 0.95 

# --- Lifecycle Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the Singleton Engine
    engine = PINNEngine()
    # In production, load from env: os.getenv("MODEL_PATH")
    # engine.load_model("assets/models/pinn_traced.pt") 
    print("Active Energy Router: ONLINE")
    yield
    print("Active Energy Router: OFFLINE")

# --- App Definition ---
app = FastAPI(
    title="Decode Energy Router",
    version="1.0.0",
    lifespan=lifespan
)

# --- Dependency Injection ---
def get_engine() -> PINNEngine:
    return PINNEngine()

# --- Routes ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "physics-informed"}

@app.post("/predict/solar-state", response_model=GridPrediction)
async def predict_grid_state(payload: SolarStateInput, engine: PINNEngine = Depends(get_engine)):
    """
    Real-time inference endpoint.
    Receives: 24h Telemetry Window
    Returns: Next-Step Physics State (GHI)
    """
    try:
        prediction = engine.predict(payload.telemetry_window)
        
        return GridPrediction(
            ghi_forecast=prediction,
            confidence_score=0.98 
        )
        
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not ready")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  
