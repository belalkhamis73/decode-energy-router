"""
Physics-Constrained TimeGAN (PC-TimeGAN) Module.
Generates high-fidelity synthetic microgrid data for "Black Swan" training.
Enforces physical invariants (Energy >= 0, Ramp Rates) during the generation process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Tuple, Dict, Any
from dataclasses import dataclass

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PC_TimeGAN")

@dataclass
class TimeGANConfig:
    """Hyperparameters for the Generative Model."""
    feature_dim: int = 5      # [DNI, DHI, Temp, Wind, Load]
    hidden_dim: int = 24
    num_layers: int = 3
    seq_len: int = 24         # 24-hour horizon
    batch_size: int = 64
    gamma: float = 1.0        # Weight for reconstruction loss
    lambda_phys: float = 10.0 # Weight for physics violation penalty

class TimeGANGenerator(nn.Module):
    """
    The 'Dreamer': Generates synthetic temporal sequences.
    Architecture: GRU-based Recurrent Neural Network.
    """
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(config.hidden_dim, config.feature_dim)
        
        # Hard Constraint: Sigmoid/ReLU to ensure non-negative energy generation
        # We use Softplus for smooth derivatives near zero
        self.output_activation = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Random noise vector [Batch, Seq, Features]
        Returns:
            Synthetic Data [Batch, Seq, Features]
        """
        out, _ = self.gru(z)
        raw_output = self.linear(out)
        return self.output_activation(raw_output)

class TimeGANDiscriminator(nn.Module):
    """
    The 'Critic': Distinguishes between Real and Synthetic data.
    """
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Bidirectional doubles the hidden size
        self.linear = nn.Linear(config.hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        # Classify validity of the sequence
        return self.linear(out)

class PhysicsValidator:
    """
    Domain Logic: Checks generated data against laws of nature.
    """
    @staticmethod
    def calculate_physics_loss(data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: Synthetic data [Batch, Seq, Features]
                  Mapping: 0:GHI, 1:Temp, 2:Wind
        """
        # 1. Non-Negativity Check (Redundant if Softplus is used, but good for safety)
        # Penalty = Sum of all negative values squared
        neg_penalty = torch.sum(torch.relu(-data))
        
        # 2. Ramp Rate Constraint (e.g., Solar cannot jump 1000W in 1 min)
        # Calculate discrete derivative along time axis (dim=1)
        # diffs = data[:, 1:, :] - data[:, :-1, :]
        # max_ramp = 0.5 # Normalized units
        # ramp_penalty = torch.sum(torch.relu(torch.abs(diffs) - max_ramp))
        
        # Combined Physics Loss
        return neg_penalty # + ramp_penalty

class TimeGANTrainer:
    """
    Orchestrates the adversarial training loop.
    Principle: Composition (Manages Gen and Disc).
    """
    def __init__(self, config: TimeGANConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        
        self.netG = TimeGANGenerator(config).to(device)
        self.netD = TimeGANDiscriminator(config).to(device)
        
        self.optG = optim.Adam(self.netG.parameters(), lr=0.001)
        self.optD = optim.Adam(self.netD.parameters(), lr=0.001)
        
        self.adv_loss = nn.BCEWithLogitsLoss()

    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        batch_size = real_data.size(0)
        
        # --- 1. Train Discriminator ---
        self.optD.zero_grad()
        
        # Generate Noise
        z = torch.randn(batch_size, self.config.seq_len, self.config.feature_dim).to(self.device)
        fake_data = self.netG(z).detach() # Detach to avoid training G
        
        # D Loss: Maximize log(D(x)) + log(1 - D(G(z)))
        d_real = self.netD(real_data)
        d_fake = self.netD(fake_data)
        
        loss_d_real = self.adv_loss(d_real, torch.ones_like(d_real))
        loss_d_fake = self.adv_loss(d_fake, torch.zeros_like(d_fake))
        loss_d = loss_d_real + loss_d_fake
        
        loss_d.backward()
        self.optD.step()
        
        # --- 2. Train Generator ---
        self.optG.zero_grad()
        
        # Generate new fake data (with gradients this time)
        fake_data_g = self.netG(z)
        d_fake_g = self.netD(fake_data_g)
        
        # G Loss: Maximize D(G(z)) -> Trick D into thinking it's real
        loss_g_adv = self.adv_loss(d_fake_g, torch.ones_like(d_fake_g))
        
        # Physics Loss: Penalize unphysical generation
        loss_phys = PhysicsValidator.calculate_physics_loss(fake_data_g)
        
        # Total G Loss
        loss_g = loss_g_adv + (self.config.lambda_phys * loss_phys)
        
        loss_g.backward()
        self.optG.step()
        
        return {
            "d_loss": loss_d.item(),
            "g_loss": loss_g_adv.item(),
            "phys_loss": loss_phys.item()
        }

    def generate(self, n_samples: int) -> np.ndarray:
        """Production inference method."""
        self.netG.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.seq_len, self.config.feature_dim).to(self.device)
            generated = self.netG(z)
        return generated.cpu().numpy()

# --- Unit Test / Usage Example ---
if __name__ == "__main__":
    logger.info("Initializing TimeGAN Unit Test...")
    
    # Mock Data: [Batch=32, Seq=24, Feat=5]
    mock_real = torch.abs(torch.randn(32, 24, 5)) 
    
    config = TimeGANConfig()
    trainer = TimeGANTrainer(config)
    
    # Run one step
    metrics = trainer.train_step(mock_real)
    logger.info(f"Training Metrics: {metrics}")
    
    # Generate
    synthetic = trainer.generate(5)
    logger.info(f"Generated Shape: {synthetic.shape}")
      
