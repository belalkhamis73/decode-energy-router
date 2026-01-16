import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time

# Import the architecture defined above
from ml_models.architectures.hybrid_model import HybridLSTMPINN

# Configure Logging (12-Factor App: Treat logs as event streams)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PINN_Trainer")

@dataclass
class TrainingConfig:
    """Immutable configuration object."""
    input_dim: int = 5
    hidden_dim: int = 64
    output_dim: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    w_data: float = 1.0       # MSE Weight
    w_physics: float = 0.1    # Physics Residual Weight (Annealing recommended)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PhysicsLoss(nn.Module):
    """
    The 'Judge' of the system.
    Calculates the residual of the governing physical equation:
    GHI ≈ DNI * cos(zenith) + DHI
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_ghi: torch.Tensor, physics_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_ghi: The model's prediction [Batch, 1]
            physics_inputs: Tensor containing [DNI, DHI, Cos_Zenith] for the target hour.
        """
        # Unpack physics parameters (Assuming specific column order from DataPipeline)
        # Col 0: DNI, Col 1: DHI, Col 2: Cos_Zenith (Variable indices depend on your specific Dataset class)
        dni = physics_inputs[:, 0]
        dhi = physics_inputs[:, 1]
        cos_theta = physics_inputs[:, 2]

        # The Theoretical Equation (Domain Knowledge)
        # Note: In normalized space, this relationship holds if scaler is linear. 
        # Ideally, we denormalize inside loss or operate in normalized physics space.
        theoretical_ghi = (dni * cos_theta) + dhi
        
        # The Residual: How far is the prediction from the physical law?
        # We penalize the difference between the NN's output and the algebraic sum
        residual = pred_ghi.squeeze() - theoretical_ghi
        
        return torch.mean(residual ** 2)

class PINNTrainer:
    """
    Manages the training lifecycle.
    Principle: Dependency Injection (Model and Config passed in).
    """
    def __init__(self, model: nn.Module, config: TrainingConfig, optimizer: Optional[optim.Optimizer] = None):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Loss Components
        self.data_criterion = nn.MSELoss()
        self.physics_criterion = PhysicsLoss()
        
        logger.info(f"Trainer initialized on device: {self.config.device}")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0

        for x, y_true, phys_inputs in dataloader:
            # Move data to device
            x = x.to(self.config.device)
            y_true = y_true.to(self.config.device)
            phys_inputs = phys_inputs.to(self.config.device)

            self.optimizer.zero_grad()

            # 1. Forward Pass
            y_pred = self.model(x)

            # 2. Compute Losses
            # Data Loss: Fit the training distribution
            l_data = self.data_criterion(y_pred, y_true)
            
            # Physics Loss: Enforce consistency with NREL physics
            # Note: We use the physics_inputs (DNI, CosTheta) to validate the output
            l_phys = self.physics_criterion(y_pred, phys_inputs)

            # Composite Loss
            loss = (self.config.w_data * l_data) + (self.config.w_physics * l_phys)

            # 3. Backward Pass
            loss.backward()
            
            # Gradient Clipping (Stability for LSTM)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_data_loss += l_data.item()
            total_phys_loss += l_phys.item()

        avg_loss = total_loss / len(dataloader)
        return {
            "loss": avg_loss,
            "data_loss": total_data_loss / len(dataloader),
            "physics_loss": total_phys_loss / len(dataloader)
        }

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop with 'Fail Fast' validation checks."""
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            metrics = self.train_epoch(train_loader)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.config.epochs} | Loss: {metrics['loss']:.6f} | Phys_Res: {metrics['physics_loss']:.6f}")
                
                # Fail Fast: If physics loss explodes, stop immediately
                if metrics['physics_loss'] > 1e3:
                    logger.error("Physics loss divergence detected. Aborting training.")
                    break

        duration = time.time() - start_time
        logger.info(f"Training completed in {duration:.2f} seconds.")

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Mock Data for testing logic independently (Unit Testing approach)
    # In production, import SolarDataset from data_pipeline
    
    logger.info("Starting Unit Test for PINN Trainer...")
    
    # 1. Configuration
    conf = TrainingConfig(epochs=5, w_physics=0.5)
    
    # 2. Instantiate Model
    pinn_model = HybridLSTMPINN(input_dim=conf.input_dim, hidden_dim=conf.hidden_dim, output_dim=conf.output_dim)
    
    # 3. Mock Dataloader (Batch=2, Seq=24, Feat=5)
    mock_x = torch.randn(10, 24, 5) 
    mock_y = torch.randn(10, 1)
    # Phys inputs: DNI, DHI, CosTheta (3 features)
    mock_phys = torch.abs(torch.randn(10, 3)) 
    
    dataset = torch.utils.data.TensorDataset(mock_x, mock_y, mock_phys)
    loader = DataLoader(dataset, batch_size=2)
    
    # 4. Run Train
    trainer = PINNTrainer(pinn_model, conf)
    trainer.fit(loader)
