"""
Deep Operator Network (DeepONet) Architecture.
Learns the solution operator for power flow equations, enabling sub-millisecond 
active energy routing by mapping input load profiles (Branch) to state trajectories (Trunk).
"""

import torch
import torch.nn as nn
from typing import Tuple

class BranchNet(nn.Module):
    """
    Encodes the input function space (e.g., The 24-hour Load Profile of the Microgrid).
    "Sensors" are the discretization points of the input function.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Tanh is standard for operator learning
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Number_of_Sensors]
        return self.net(x)

class TrunkNet(nn.Module):
    """
    Encodes the coordinate space (e.g., Time 't' or Location 'x' on the grid).
    Queries the operator at specific points in the domain.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [Batch, Domain_Dimension] (e.g., Time)
        return self.net(t)

class DeepONet(nn.Module):
    """
    Unstacked DeepONet implementation.
    Output G(u)(y) = Sum(Branch_k(u) * Trunk_k(y)) + Bias
    
    Principles:
    - Composition: Builds complex operator from simple Branch/Trunk networks.
    - Universal Approximation: Can approximate any continuous non-linear operator.
    """
    def __init__(self, 
                 sensor_dim: int = 24,    # Input function resolution (e.g., 24 hours)
                 domain_dim: int = 1,     # Query domain (1 for Time, 2 for Space-Time)
                 hidden_dim: int = 64,    # Width of hidden layers
                 latent_dim: int = 64):   # Dimension of the dot product space
        super().__init__()
        
        self.branch = BranchNet(sensor_dim, hidden_dim, latent_dim)
        self.trunk = TrunkNet(domain_dim, hidden_dim, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        # Initialize weights for operator learning stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input function samples [Batch, Sensor_Dim] (The Load Profile)
            y: Query locations [Batch, Domain_Dim] (The Time 't' to predict Voltage)
            
        Returns:
            Operator output G(u)(y) [Batch, 1]
        """
        # 1. Encode Input Function
        b_out = self.branch(u) # [Batch, Latent_Dim]
        
        # 2. Encode Domain Coordinates
        t_out = self.trunk(y)  # [Batch, Latent_Dim]
        
        # 3. Dot Product (The "Crossing" of Information)
        # Element-wise product followed by sum across latent dimension
        # Ensures interaction between input function and query location
        dot_prod = torch.sum(b_out * t_out, dim=1, keepdim=True)
        
        return dot_prod + self.bias

# --- Unit Test ---
if __name__ == "__main__":
    # Simulate a scenario:
    # We have 10 samples (Batch=10).
    # Each sample is a 24-hour load profile (Sensor_Dim=24).
    # We want to query the grid state at a specific time 't' (Domain_Dim=1).
    
    batch_size = 10
    model = DeepONet(sensor_dim=24, domain_dim=1)
    
    u_dummy = torch.randn(batch_size, 24) # Random load profiles
    y_dummy = torch.rand(batch_size, 1)   # Random query times [0, 1]
    
    output = model(u_dummy, y_dummy)
    print(f"DeepONet Output Shape: {output.shape}") # Should be [10, 1]
    
