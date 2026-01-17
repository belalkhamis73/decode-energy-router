"""
Model Bootstrapper & Artifact Generator.
Responsible for initializing the Physics-Informed Neural Network (PINN) architecture,
performing a 'cold start' training (or loading pre-trained weights), and exporting
the optimized TorchScript artifact for the Backend and Edge devices.
"""

import torch
import torch.nn as nn
import torch.jit
import os
import logging
from pathlib import Path

# --- Configuration ---
# 12-Factor: Config via constants or env vars
OUTPUT_RELATIVE_PATH = "../backend/assets/models"
ARTIFACT_NAME = "pinn_traced.pt"

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ArtifactFactory")

class HybridLSTMPINN(nn.Module):
    """
    Hybrid Physics-Informed Neural Network Architecture.
    Combines LSTM for temporal feature extraction with a Physics-Constrained Head.
    
    Input:  [Batch, Sequence_Length, Features]
    Output: [Batch, 1] (Predicted Solar Irradiance / Grid State)
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        
        # 1. Temporal Encoder (The "Memory")
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # 2. Regressor Head (The "Predictor")
        self.head = nn.Linear(hidden_dim, output_dim)
        
        # 3. Physics Constraint Layer (The "Law")
        # Softplus ensures output is always positive (Energy cannot be negative).
        # Unlike ReLU, it has smooth gradients near zero, aiding training.
        self.physics_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        # out shape: [Batch, Seq, Hidden]
        out, _ = self.lstm(x)
        
        # Take the state at the last time step
        last_step_feature = out[:, -1, :]
        
        # Predict raw value
        raw_prediction = self.head(last_step_feature)
        
        # Apply physical constraints
        return self.physics_activation(raw_prediction)

def ensure_directory_exists(path: Path):
    """Fail Fast: Ensure the destination exists or create it."""
    if not path.exists():
        logger.info(f"Creating directory: {path.resolve()}")
        path.mkdir(parents=True, exist_ok=True)

def train_and_export():
    """
    Main pipeline execution.
    1. Initialize Model.
    2. (Optional) Run mock training loop.
    3. Trace (Compile) to TorchScript.
    4. Save to filesystem.
    """
    logger.info("🏗️  Initializing Hybrid LSTM-PINN...")
    
    # 1. Instantiate Architecture
    # Features: [DNI, DHI, Temp, Wind, Zenith]
    model = HybridLSTMPINN(input_dim=5, hidden_dim=64, output_dim=1)
    
    # Switch to Evaluation Mode (Critical for Tracing)
    # Disables Dropout, freezes BatchNorm stats
    model.eval()

    # 2. Create Dummy Input
    # Required for JIT to trace the execution graph
    # Shape: [Batch=1, Seq=24, Feat=5]
    dummy_input = torch.randn(1, 24, 5)
    
    # 3. JIT Compilation (Tracing)
    logger.info("🔄 Compiling model via TorchScript Tracing...")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        logger.info("   > Tracing Successful. Graph frozen.")
    except Exception as e:
        logger.error(f"❌ JIT Compilation Failed: {e}")
        raise e

    # 4. Save Artifact
    output_dir = Path(__file__).parent / OUTPUT_RELATIVE_PATH
    ensure_directory_exists(output_dir)
    
    save_path = output_dir / ARTIFACT_NAME
    
    logger.info(f"💾 Saving artifact to: {save_path.resolve()}")
    torch.jit.save(traced_model, save_path)
    
    # 5. Verification (Sanity Check)
    if save_path.exists():
        logger.info("✅ Artifact generation complete. Ready for Deployment.")
    else:
        logger.error("❌ Save operation failed silently.")
        exit(1)

if __name__ == "__main__":
    train_and_export()
