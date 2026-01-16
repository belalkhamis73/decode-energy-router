import torch
import torch.nn as nn
from typing import Tuple

class TimeSeriesEncoder(nn.Module):
    """
    Encodes temporal dynamics from weather and grid sensors.
    Component 1 of the Hybrid Architecture.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, Features]
        Returns:
            Latent representation of the last time step [Batch, Hidden_Dim]
        """
        out, _ = self.lstm(x)
        return out[:, -1, :]  # We only need the state at t_final for forecasting

class StateEstimatorHead(nn.Module):
    """
    Maps latent temporal features to physical grid states.
    Component 2 of the Hybrid Architecture.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Tanh activation is critical for PINNs to ensure non-vanishing higher-order derivatives
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization to stabilize gradients in physics-informed training."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridLSTMPINN(nn.Module):
    """
    The composite 'Brain' of the Digital Twin.
    Combines LSTM memory with a Physics-Informed Dense head.
    
    Inputs: [DNI, DHI, Temp, Wind, Cos_Zenith] over T hours.
    Outputs: [GHI_Predicted] (and potentially Voltage/Freq in future iterations).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        
        # Composition over Inheritance
        self.encoder = TimeSeriesEncoder(input_dim, hidden_dim, num_layers, dropout=0.2)
        self.estimator = StateEstimatorHead(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_state = self.encoder(x)
        physical_prediction = self.estimator(latent_state)
        return physical_prediction
