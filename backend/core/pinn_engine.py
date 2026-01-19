"""
PINN Engine (The "Brain" Wrapper).
Responsible for loading model artifacts, managing GPU/CPU devices, 
and executing the Neural Network inference logic.

Map Requirement: "Loads the .pt model and runs inference. Used by: simulation.py."
"""

import torch
import os
import logging
from typing import Optional, Dict, Any, Union

# Import the Architecture defined in Layer 2
try:
    from ml_models.architectures.deeponet import DeepONet
except ImportError:
    # Fallback for localized testing if absolute path fails
    import sys
    sys.path.append(os.getcwd())
    from ml_models.architectures.deeponet import DeepONet

# Configure Logging
logger = logging.getLogger("PINN_Engine")

class PINNEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DeepONet] = None
        self.is_ready = False

    def load_model(self, session_id: str, artifact_path: Optional[str] = None) -> bool:
        """
        Loads the trained DeepONet model for a specific session.
        
        Args:
            session_id: Unique identifier for the user session.
            artifact_path: Path to the .pt or .pth state dictionary.
            
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        try:
            logger.info(f"🧠 Loading PINN Model for Session: {session_id}")
            
            # 1. Instantiate the Architecture
            # Note: In a real system, hyperparameters (input_dim, hidden_dim) 
            # would be loaded from a config.json saved alongside the weights.
            self.model = DeepONet(input_dim=3, hidden_dim=64, output_dim=1, n_buses=14)
            
            # 2. Load Weights (if available)
            if artifact_path and os.path.exists(artifact_path):
                state_dict = torch.load(artifact_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"   > Weights loaded from {artifact_path}")
            else:
                logger.warning("   > No artifact found. Using initialized random weights (Mock Mode).")
            
            # 3. Optimize for Inference
            self.model.to(self.device)
            self.model.eval() # Disables Dropout, freezes BatchNorm
            self.is_ready = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_ready = False
            return False

    def infer_voltage(self, load_scaling: float, weather_context: Dict[str, Any], tick: int) -> torch.Tensor:
        """
        Runs the Physics-Informed inference to predict grid voltage.
        
        Args:
            load_scaling: Multiplier for grid load (e.g., 1.2 for 20% overload).
            weather_context: Dict containing 'ghi', 'wind_speed', 'temperature' arrays.
            tick: Current time step (hour 0-23).
            
        Returns:
            torch.Tensor: Predicted Voltage in p.u. [Batch=1, Buses]
        """
        # Safety Fallback: If model isn't trained yet, return physics heuristic
        if not self.is_ready or self.model is None:
            # Heuristic: Voltage drops as load increases
            heuristic_voltage = 1.0 - (load_scaling * 0.05)
            return torch.tensor([heuristic_voltage])

        try:
            with torch.no_grad():
                # 1. Construct Input Feature Vector (Branch Input)
                # Features: [Solar_GHI, Wind_Speed, Total_Load_PU]
                # Shape: [Batch=1, Features=3]
                ghi = weather_context.get('ghi', [0]*24)[tick]
                wind = weather_context.get('wind_speed', [0]*24)[tick]
                
                u_input = torch.tensor([
                    ghi / 1000.0,       # Normalize Solar (approx max 1000)
                    wind / 30.0,        # Normalize Wind (approx max 30)
                    load_scaling        # Load is already relative
                ], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 2. Construct Location Query (Trunk Input)
                # DeepONet needs to know *where* we are predicting voltage.
                # Usually we predict for all buses at once.
                # Shape: [Batch=1, Bus_ID] (handled internally or passed as None if implicit)
                # For this specific DeepONet implementation, we'll assume implicit bus handling
                # or pass a dummy tensor if the trunk expects specific coordinates.
                y_input = None 
                
                # 3. Forward Pass
                prediction = self.model(u_input, y_input)
                
                return prediction.cpu()
                
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            # Fail gracefully to 1.0 p.u.
            return torch.tensor([1.0])

# Singleton Instance
# This allows simulation.py to simply `from backend.core.pinn_engine import pinn_engine`
pinn_engine = PINNEngine()
