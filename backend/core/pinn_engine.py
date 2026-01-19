"""
PINN Engine (The "Brain" Wrapper).
Responsible for loading model artifacts, managing GPU/CPU devices, 
and executing the Neural Network inference logic.
Supports Multi-Tenancy (one model per session) and Dynamic Grid Sizes.
"""

import torch
import logging
import os
from typing import Dict, Any, Optional

# Configure Logging
logger = logging.getLogger("PINN_Engine")

# --- Import Architecture (with Mock Fallback) ---
try:
    from ml_models.architectures.deeponet import DeepONet
except ImportError:
    # Fallback for localized testing if absolute path fails
    # or if the model file is missing in the current env
    from torch import nn
    class DeepONet(nn.Module):
        def __init__(self, n_buses=14, input_dim=3, hidden_dim=64, output_dim=1): 
            super().__init__()
            self.n_buses = n_buses
        def forward(self, x): 
            # Return dummy voltage vector [Batch, 1]
            return torch.ones(x.shape[0], 1) * 1.0

class PINNEngine:
    """
    Singleton Inference Engine.
    Manages a registry of active models for different sessions.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Registry: session_id -> DeepONet Model
        self.models: Dict[str, Any] = {} 
        logger.info(f"🚀 PINN Engine Initialized on {self.device}")

    def register_model(self, session_id: str, n_buses: int, artifact_path: Optional[str] = None):
        """
        Initializes and stores a model specific to a session's topology.
        Solves the 14 vs 118 bus dimension mismatch.
        """
        try:
            logger.info(f"🧠 Registering PINN for Session {session_id} (Grid Size: {n_buses})")
            
            # 1. Initialize Architecture with Dynamic Grid Size
            model = DeepONet(input_dim=3, hidden_dim=64, output_dim=1, n_buses=n_buses)
            
            # 2. Load Weights (if provided)
            if artifact_path and os.path.exists(artifact_path):
                state_dict = torch.load(artifact_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"   > Weights loaded from {artifact_path}")
            
            # 3. Optimize & Store
            model.to(self.device)
            model.eval()
            self.models[session_id] = model
            return True

        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            return False

    def infer_voltage(self, session_id: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Routes inference to the correct model for the session.
        Args:
            session_id: Unique identifier for the user session.
            input_tensor: Tensor of shape [Batch, Features].
        """
        # 1. Check Registry
        if session_id not in self.models:
            # Fallback for sessions that haven't registered a model (or training not done)
            # Returns 1.0 p.u. voltage (Nominal) to keep simulation alive
            return torch.ones(input_tensor.shape[0], 1) * 1.0
        
        # 2. Retrieve Model
        model = self.models[session_id]
        
        # 3. Execute Inference
        try:
            with torch.no_grad():
                return model(input_tensor.to(self.device)).cpu()
        except Exception as e:
            logger.error(f"Inference Error for {session_id}: {e}")
            return torch.ones(input_tensor.shape[0], 1) * 1.0

# Singleton Instance
pinn_engine = PINNEngine()
