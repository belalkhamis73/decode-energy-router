import torch
import logging
from pathlib import Path
from typing import Optional, List
from ml_models.architectures.hybrid_model import HybridLSTMPINN

logger = logging.getLogger(__name__)

class PINNEngine:
    """
    Singleton Inference Engine.
    Manages the lifecycle of the Physics-Informed Neural Network.
    """
    _instance: Optional['PINNEngine'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PINNEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[HybridLSTMPINN] = None
        self._initialized = True
        logger.info(f"PINN Engine initialized on {self.device}")

    def load_model(self, model_path: str, input_dim: int = 5, hidden_dim: int = 64) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        try:
            # Reconstruct architecture
            self.model = HybridLSTMPINN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1)
            
            # Load weights
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval() # Freeze dropout/batchnorm for inference
            
            logger.info("Physics-Informed weights loaded successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load PINN model: {e}")
            raise

    def predict(self, features: List[List[float]]) -> float:
        """
        Executes the forward pass.
        Args:
            features: 2D List [Seq_Len, Features] matching the LSTM window.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        with torch.no_grad():
            tensor_in = torch.tensor([features], dtype=torch.float32).to(self.device)
            prediction = self.model(tensor_in)
            return prediction.item()
                                        
