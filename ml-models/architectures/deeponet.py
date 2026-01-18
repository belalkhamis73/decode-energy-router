"""
Deep Operator Network (DeepONet) Architecture.
Learns the solution operator for power flow equations (V = G(P, Q)), enabling 
sub-millisecond active energy routing by mapping input Load/Weather profiles (Branch) 
to Grid State trajectories at specific Bus locations (Trunk).

Alterations for SaaS:
- Added 'n_buses' to __init__ for dynamic topology sizing (IEEE 14 vs 118).
- Added Embedding layer to TrunkNet to handle discrete grid locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BranchNet(nn.Module):
    """
    Encodes the Input Function Space 'u'.
    Context: The environmental and operational conditions (Weather + Load).
    Input: [Batch, Sensor_Dim] (e.g., Solar Irradiance, Wind Speed, Total Load).
    Output: [Batch, Latent_Dim] (The encoded 'Context' vector).
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh is standard for operator learning (smooth gradients)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TrunkNet(nn.Module):
    """
    Encodes the Domain Space 'y'.
    Context: The discrete locations on the grid (Bus IDs).
    
    Alteration: Uses an Embedding layer instead of continuous coordinates
    to map discrete Bus IDs (0...N) to a dense vector space.
    """
    def __init__(self, n_buses: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        # Dynamic Resizing: Maps discrete Bus IDs (0 to n_buses-1) to dense vectors
        # This allows the same architecture code to handle IEEE 14 or IEEE 118
        self.bus_embedding = nn.Embedding(num_embeddings=n_buses, embedding_dim=hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, bus_ids: torch.Tensor) -> torch.Tensor:
        # bus_ids shape: [Batch, N_Query_Buses]
        
        # 1. Embed discrete IDs -> Dense Vectors
        # Shape: [Batch, N_Query_Buses, Hidden_Dim]
        x = self.bus_embedding(bus_ids)
        
        # 2. Process through MLP
        # Shape: [Batch, N_Query_Buses, Latent_Dim]
        return self.net(x)

class DeepONet(nn.Module):
    """
    The Operator: G(u)(y).
    Computes the dot product between the Context (Branch) and the Location (Trunk).
    """
    def __init__(self, 
                 input_dim: int = 3,     # [Solar, Wind, Load]
                 hidden_dim: int = 64, 
                 output_dim: int = 1,    # Voltage Magnitude
                 n_buses: int = 14,      # Default to IEEE 14, but dynamic
                 latent_dim: int = 64):
        super().__init__()
        
        self.n_buses = n_buses
        
        # 1. The Branch (Weather/Load Encoder)
        self.branch = BranchNet(input_dim, hidden_dim, latent_dim)
        
        # 2. The Trunk (Grid Location Encoder)
        # Note: This is where we injected the 'n_buses' alteration
        self.trunk = TrunkNet(n_buses, hidden_dim, latent_dim)
        
        # 3. Bias term for the final regression
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, bus_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input conditions [Batch, Input_Dim] 
               (e.g., current solar/wind/load)
            bus_ids: The specific buses we want to check [Batch, N_Query_Buses]
               (e.g., checking all 118 buses at once)
               
        Returns:
            Prediction: [Batch, N_Query_Buses] (Voltage at these buses)
        """
        # 1. Encode Condition (Branch)
        # Shape: [Batch, Latent_Dim]
        b_out = self.branch(u) 
        
        # 2. Encode Locations (Trunk)
        # Shape: [Batch, N_Query_Buses, Latent_Dim]
        t_out = self.trunk(bus_ids) 
        
        # 3. Operator Composition (Dot Product)
        # We need to broadcast b_out to match the number of buses in t_out
        # b_out expanded: [Batch, 1, Latent_Dim]
        # t_out:          [Batch, N, Latent_Dim]
        
        # Element-wise product + Sum over latent dimension
        # Result: [Batch, N_Query_Buses]
        prediction = torch.sum(b_out.unsqueeze(1) * t_out, dim=2) + self.bias
        
        return prediction

# --- Unit Test (Verification of Alterations) ---
if __name__ == "__main__":
    print("🔬 DeepONet Architecture Unit Test")
    
    # Simulation Parameters
    BATCH_SIZE = 5
    N_BUSES_IEEE118 = 118  # Testing the "Biggest Dataset" requirement
    INPUT_FEATURES = 3     # Solar, Wind, Load
    
    # 1. Instantiate Model with Dynamic Bus Count
    model = DeepONet(
        input_dim=INPUT_FEATURES, 
        n_buses=N_BUSES_IEEE118
    )
    print(f"   ✅ Model Initialized for {model.n_buses} Buses (IEEE 118 Mode)")
    
    # 2. Create Dummy Data
    # Weather/Load context
    dummy_weather = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    
    # Query: We want to predict voltage for ALL 118 buses at once
    # Shape: [Batch, 118]
    dummy_bus_ids = torch.arange(N_BUSES_IEEE118).unsqueeze(0).repeat(BATCH_SIZE, 1)
    
    # 3. Forward Pass
    try:
        output = model(dummy_weather, dummy_bus_ids)
        
        # 4. Check Shapes
        expected_shape = (BATCH_SIZE, N_BUSES_IEEE118)
        assert output.shape == expected_shape
        
        print(f"   ✅ Forward Pass Successful.")
        print(f"      Input: {dummy_weather.shape} (Weather Context)")
        print(f"      Query: {dummy_bus_ids.shape} (All Buses)")
        print(f"      Output: {output.shape} (Voltage Prediction)")
        
    except Exception as e:
        print(f"   ❌ Test Failed: {e}")
