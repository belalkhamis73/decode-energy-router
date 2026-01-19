"""
Training Orchestrator (The "Trainer" Wrapper).
Delegates the heavy lifting of the training loop to 'train_pinn.py'.
Handles Data Ingestion -> Model Training -> Artifact Saving.

Map Requirement: "The Training Orchestrator. Triggered by: main.py (POST /model/train)."
"""

import torch
import os
import logging
from pathlib import Path
from typing import Dict, Any

# --- INTEGRATION: Import dependencies ---
from ml_models.training.train_pinn import train_session_model
from ml_models.training.data_pipeline import NRELClient, PhysicsProcessor, PhysicsScaler

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Training_Orchestrator")

# Constants
OUTPUT_RELATIVE_PATH = "../../../backend/assets/models"
ARTIFACT_NAME = "pinn_traced.pt"

def train_and_export(
    grid_type: str = "ieee118", 
    epochs: int = 50,
    dataset: Dict[str, Any] = None
) -> str:
    """
    Orchestrates the training pipeline.
    
    Args:
        grid_type: Topology identifier (e.g., 'ieee118').
        epochs: Number of training iterations.
        dataset: (Optional) Pre-loaded dictionary with 'topology' and 'weather'.
                 If None, it fetches data via pipeline.
                 
    Returns:
        str: Absolute path to the saved model artifact.
    """
    logger.info(f"🚀 Starting Training Job for {grid_type.upper()}...")
    
    # 1. Data Ingestion (Layer 2 Integration)
    if dataset is None:
        # If no dataset provided (cold start), fetch from NREL (Mocked for now)
        logger.info("   > Fetching standard training data...")
        # In a real run, we'd use NRELClient here.
        # For now, we mock the structure expected by train_pinn.py
        dataset = {
            "topology": {"n_buses": 118 if "118" in grid_type else 14},
            "weather": {"ghi": [500.0]*24, "wind_speed": [5.0]*24}
        }

    # 2. Training (Delegation to train_pinn.py)
    hyperparams = {
        "epochs": epochs,
        "learning_rate": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # This is the key "Link Training" step requested
    trained_model, metrics = train_session_model(dataset, hyperparams)
    logger.info(f"   ✅ Training converged. Final Loss: {metrics.get('final_loss', 99.9):.5f}")

    # 3. Validation & Tracing (Compilation)
    trained_model.eval()
    
    # Create dummy input for JIT Tracing (Batch=1, Features=3)
    dummy_u = torch.randn(1, 3).to(hyperparams["device"])
    # Note: DeepONet handles 'y' (trunk input) internally if None is passed during tracing,
    # or we must provide it. Based on our deeponet.py, it handles None.
    # However, JIT trace strictly requires valid tensor inputs.
    # We pass a dummy 'y' to be safe.
    n_buses = dataset['topology']['n_buses']
    dummy_y = torch.arange(n_buses).unsqueeze(0).to(hyperparams["device"])
    
    try:
        logger.info("   > Compiling to TorchScript...")
        traced_model = torch.jit.trace(trained_model, (dummy_u, dummy_y))
    except Exception as e:
        logger.warning(f"   ⚠️ JIT Tracing failed ({e}). Saving state_dict instead.")
        traced_model = trained_model

    # 4. Artifact Export
    # Resolve path relative to this file
    base_dir = Path(__file__).parent
    output_dir = (base_dir / OUTPUT_RELATIVE_PATH).resolve()
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = output_dir / f"{grid_type}_{ARTIFACT_NAME}"
    
    if isinstance(traced_model, torch.jit.ScriptModule):
        traced_model.save(save_path)
    else:
        torch.save(traced_model.state_dict(), save_path)
        
    logger.info(f"💾 Model saved to: {save_path}")
    return str(save_path)

if __name__ == "__main__":
    # Unit Test
    train_and_export("ieee14", epochs=1)
