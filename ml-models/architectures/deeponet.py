"""
Deep Operator Network (DeepONet) Architecture.
Learns the solution operator for power flow equations (V = G(P, Q)).

Integration Update:
- Physically embeds the 'HardConstraintProjection' layer as the final output block.
- Ensures all predictions strictly adhere to voltage limits (0.9 - 1.1 p.u.) during inference.

Map Requirement: "architectures/projection_layer.py: Inside: deeponet.py."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# --- INTEGRATION: Import Projection Layer ---
try:
    from ml_models.architectures.projection_layer import HardConstraintProjection
except ImportError:
    # Fallback for localized testing if absolute path fails
    import sys
    import os
    sys.path.append(os.getcwd())
    from ml_models.architectures.projection_layer import HardConstraintProjection

class DeepONet(nn.Module):
    """
    Physics-Informed DeepONet with Integrated Safety Layer.
    
    Structure:
    1. Branch Net: Encodes Input Function u (Weather/Load Context).
    2. Trunk Net: Encodes Output Location y (Grid Bus IDs).
    3. Dot Product: Merges Context + Location.
    4. Bias: System-wide offset.
    5. Projection: Hard constraints on Voltage (Safety Valve).
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 1, n_buses: int = 14):
        super().__init__()
        self.n_buses = n_buses
        
        # 1. Branch Net (Context Encoder)
        # Input: [Solar, Wind, Load] -> Latent Vector
        self.branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Trunk Net (Location Encoder)
        # Input: Bus ID or Coordinate -> Latent Vector
        # We use an Embedding layer for discrete bus locations
        self.trunk_embedding = nn.Embedding(n_buses, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. System Bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 4. SAFETY INTEGRATION: Projection Layer
        # Enforces V_min <= V <= V_max
        self.projection = HardConstraintProjection(num_nodes=n_buses)

    def forward(self, u: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward Pass with Physics Enforcement.
        
        Args:
            u: Input features [Batch, Input_Dim] (Weather + Load).
            y: (Optional) Specific Bus IDs to query [Batch, Query_Size].
               If None, predicts for ALL buses in the grid topology.
               
        Returns:
            Voltage Vector [Batch, N_Buses] (Projected onto valid manifold).
        """
        batch_size = u.shape[0]
        
        # A. Branch Output (Context)
        # Shape: [Batch, Hidden]
        b_out = self.branch(u)
        
        # B. Trunk Output (Location)
        if y is None:
            # If no specific buses requested, predict for ALL buses
            # Create IDs: [0, 1, ..., N_Buses-1] repeated for batch
            bus_ids = torch.arange(self.n_buses, device=u.device).expand(batch_size, -1)
        else:
            bus_ids = y
            
        # Shape: [Batch, N_Buses, Hidden]
        t_out = self.trunk(self.trunk_embedding(bus_ids))
        
        # C. Operator Merge (Dot Product)
        # Equation: G(u)(y) = Sum(Branch_k * Trunk_k) + Bias
        # b_out: [Batch, 1, Hidden]
        # t_out: [Batch, N_Buses, Hidden]
        # Result: [Batch, N_Buses]
        raw_prediction = torch.sum(b_out.unsqueeze(1) * t_out, dim=2) + self.bias
        
        # D. PHYSICS INTEGRATION (The Requirement)
        # "Modify deeponet.py to actually contain the projection_layer.py"
        
        if self.training:
            # During training, we might want soft gradients or raw values for loss calculation
            # However, for strict physics compliance, we can project here too.
            # We return raw_prediction to let the Loss Function handle penalties,
            # OR return projected if we are doing "Projected Gradient Descent".
            # For standard PINNs, we typically return raw and punish violations in loss.
            return raw_prediction
        else:
            # INFERENCE MODE: Strict Safety
            # Apply Hard Projection to ensure output is always valid (0.9 - 1.1 p.u.)
            # We pass dummy A, b matrices as the layer handles bounds internally
            projected_prediction = self.projection(raw_prediction, None, None)
            return projected_prediction

# --- Unit Test ---
if __name__ == "__main__":
    print("🔬 Testing Integrated DeepONet + Projection...")
    
    # Setup
    model = DeepONet(n_buses=5)
    model.eval() # Enable Inference Mode (activates Projection)
    
    # Mock Input (Batch=2, Features=3)
    u_test = torch.randn(2, 3)
    
    # Forward Pass
    # The output should be clamped between 0.9 and 1.1 automatically
    v_out = model(u_test)
    
    print(f"   Output Shape: {v_out.shape}")
    print(f"   Min Voltage: {v_out.min().item():.4f} (Should be >= 0.9)")
    print(f"   Max Voltage: {v_out.max().item():.4f} (Should be <= 1.1)")
    
    assert v_out.min() >= 0.9, "❌ Lower Bound Violation!"
    assert v_out.max() <= 1.1, "❌ Upper Bound Violation!"
    print("✅ Integration Verified: Physics Projection Active.")
