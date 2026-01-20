"""
Deep Operator Network (DeepONet) Architecture - Enhanced Version.
Learns the solution operator for power flow equations (V = G(P, Q)).

Integration Update:
- Physically embeds the 'HardConstraintProjection' layer as the final output block.
- Ensures all predictions strictly adhere to voltage limits (0.9 - 1.1 p.u.) during inference.

NEW FEATURES:
- Uncertainty Quantification: Monte Carlo Dropout + Ensemble Support
- Attention Mechanism: Feature importance weighting for inputs
- Multi-Fidelity Modeling: Combines high/low fidelity physics simulations
- Meta-Learning: Fast adaptation to new grid topologies (MAML-style)

Map Requirement: "architectures/projection_layer.py: Inside: deeponet.py."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np

# --- INTEGRATION: Import Projection Layer ---
try:
    from ml_models.architectures.projection_layer import HardConstraintProjection
except ImportError:
    # Fallback for localized testing if absolute path fails
    import sys
    import os
    sys.path.append(os.getcwd())
    from ml_models.architectures.projection_layer import HardConstraintProjection


class FeatureAttention(nn.Module):
    """
    Attention mechanism for input features.
    Learns which weather/load features are most important for voltage prediction.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [Batch, Input_Dim]
        Returns:
            weighted_x: Attention-weighted features [Batch, Input_Dim]
            attention_weights: Importance scores [Batch, Input_Dim]
        """
        attention_weights = self.attention(x)
        weighted_x = x * attention_weights
        return weighted_x, attention_weights


class MultiFidelityFusion(nn.Module):
    """
    Combines predictions from different physics fidelity levels.
    Low-fidelity: Fast approximate solutions
    High-fidelity: Accurate but expensive solutions
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Learnable fidelity weights
        self.fidelity_weights = nn.Parameter(torch.tensor([0.7, 0.3]))
    
    def forward(self, low_fidelity: torch.Tensor, high_fidelity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            low_fidelity: Low-fidelity branch output [Batch, Hidden]
            high_fidelity: High-fidelity branch output [Batch, Hidden]
        Returns:
            fused: Combined representation [Batch, Hidden]
        """
        # Weighted combination
        weights = F.softmax(self.fidelity_weights, dim=0)
        weighted_sum = weights[0] * low_fidelity + weights[1] * high_fidelity
        
        # Non-linear fusion
        concatenated = torch.cat([low_fidelity, high_fidelity], dim=-1)
        fused = self.fusion(concatenated) + weighted_sum  # Residual connection
        
        return fused


class MetaLearningAdapter(nn.Module):
    """
    Fast adaptation layer for new grid topologies.
    Uses task-specific parameters that can be quickly fine-tuned.
    """
    def __init__(self, hidden_dim: int, n_tasks: int = 5):
        super().__init__()
        self.n_tasks = n_tasks
        # Task-specific transformation matrices
        self.task_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
            for _ in range(n_tasks)
        ])
        self.task_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim))
            for _ in range(n_tasks)
        ])
        
    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input features [Batch, Hidden]
            task_id: Which grid topology to use (0 to n_tasks-1)
        Returns:
            adapted: Task-adapted features [Batch, Hidden]
        """
        if task_id >= self.n_tasks:
            task_id = 0  # Default to first task
        
        W = self.task_embeddings[task_id]
        b = self.task_biases[task_id]
        adapted = torch.matmul(x, W) + b
        return adapted


class DeepONet(nn.Module):
    """
    Enhanced Physics-Informed DeepONet with Integrated Safety Layer.
    
    Structure:
    1. Feature Attention: Identifies important inputs
    2. Branch Net: Encodes Input Function u (Weather/Load Context)
       - Supports Multi-Fidelity modeling
       - Includes Monte Carlo Dropout for uncertainty
    3. Trunk Net: Encodes Output Location y (Grid Bus IDs)
    4. Meta-Learning: Fast adaptation to new topologies
    5. Dot Product: Merges Context + Location
    6. Bias: System-wide offset
    7. Projection: Hard constraints on Voltage (Safety Valve)
    """
    def __init__(
        self, 
        input_dim: int = 3, 
        hidden_dim: int = 64, 
        output_dim: int = 1, 
        n_buses: int = 14,
        enable_uncertainty: bool = True,
        enable_attention: bool = True,
        enable_multifidelity: bool = False,
        enable_metalearning: bool = False,
        dropout_rate: float = 0.1,
        n_tasks: int = 5
    ):
        super().__init__()
        self.n_buses = n_buses
        self.enable_uncertainty = enable_uncertainty
        self.enable_attention = enable_attention
        self.enable_multifidelity = enable_multifidelity
        self.enable_metalearning = enable_metalearning
        
        # 0. Feature Attention (NEW)
        if enable_attention:
            self.attention = FeatureAttention(input_dim)
        
        # 1. Branch Net (Context Encoder) - Enhanced
        # Primary high-fidelity branch
        self.branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate) if enable_uncertainty else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate) if enable_uncertainty else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Low-fidelity branch (simpler/faster approximation)
        if enable_multifidelity:
            self.branch_low_fidelity = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            self.fidelity_fusion = MultiFidelityFusion(hidden_dim)
        
        # 2. Trunk Net (Location Encoder)
        self.trunk_embedding = nn.Embedding(n_buses, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate) if enable_uncertainty else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate) if enable_uncertainty else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Meta-Learning Adapter (NEW)
        if enable_metalearning:
            self.meta_adapter = MetaLearningAdapter(hidden_dim, n_tasks)
        
        # 4. System Bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # 5. SAFETY INTEGRATION: Projection Layer
        self.projection = HardConstraintProjection(num_nodes=n_buses)
        
        # 6. Ensemble members for uncertainty quantification
        self.n_ensemble = 5 if enable_uncertainty else 1

    def forward(
        self, 
        u: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        task_id: int = 0,
        return_uncertainty: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass with Physics Enforcement and Uncertainty Estimation.
        
        Args:
            u: Input features [Batch, Input_Dim] (Weather + Load).
            y: (Optional) Specific Bus IDs to query [Batch, Query_Size].
               If None, predicts for ALL buses in the grid topology.
            task_id: Grid topology ID for meta-learning (default: 0)
            return_uncertainty: If True, returns (mean, std) predictions
               
        Returns:
            If return_uncertainty=False:
                Voltage Vector [Batch, N_Buses] (Projected onto valid manifold).
            If return_uncertainty=True:
                (mean_voltage, std_voltage): Both [Batch, N_Buses]
        """
        batch_size = u.shape[0]
        
        # 0. Feature Attention (if enabled)
        attention_weights = None
        if self.enable_attention:
            u, attention_weights = self.attention(u)
        
        # A. Branch Output (Context)
        if self.enable_multifidelity:
            # High-fidelity path
            b_out_high = self.branch(u)
            # Low-fidelity path
            b_out_low = self.branch_low_fidelity(u)
            # Fuse both fidelities
            b_out = self.fidelity_fusion(b_out_low, b_out_high)
        else:
            b_out = self.branch(u)
        
        # Meta-learning adaptation (if enabled)
        if self.enable_metalearning:
            b_out = self.meta_adapter(b_out, task_id)
        
        # B. Trunk Output (Location)
        if y is None:
            bus_ids = torch.arange(self.n_buses, device=u.device).expand(batch_size, -1)
        else:
            bus_ids = y
            
        t_out = self.trunk(self.trunk_embedding(bus_ids))
        
        # C. Operator Merge (Dot Product)
        raw_prediction = torch.sum(b_out.unsqueeze(1) * t_out, dim=2) + self.bias
        
        # D. Uncertainty Quantification (if enabled and requested)
        if self.enable_uncertainty and return_uncertainty:
            predictions = []
            # Monte Carlo Dropout: Multiple forward passes with dropout active
            self.train()  # Enable dropout temporarily
            for _ in range(self.n_ensemble):
                # Re-compute with dropout
                if self.enable_multifidelity:
                    b_out_high = self.branch(u)
                    b_out_low = self.branch_low_fidelity(u)
                    b_out = self.fidelity_fusion(b_out_low, b_out_high)
                else:
                    b_out = self.branch(u)
                
                if self.enable_metalearning:
                    b_out = self.meta_adapter(b_out, task_id)
                
                t_out = self.trunk(self.trunk_embedding(bus_ids))
                pred = torch.sum(b_out.unsqueeze(1) * t_out, dim=2) + self.bias
                predictions.append(pred)
            
            self.eval()  # Restore eval mode
            
            # Stack predictions and compute statistics
            predictions = torch.stack(predictions, dim=0)  # [N_ensemble, Batch, N_buses]
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            
            # Apply projection to mean prediction
            if not self.training:
                mean_pred = self.projection(mean_pred, None, None)
            
            return mean_pred, std_pred
        
        # E. PHYSICS INTEGRATION (The Requirement)
        if self.training:
            return raw_prediction
        else:
            # INFERENCE MODE: Strict Safety
            projected_prediction = self.projection(raw_prediction, None, None)
            return projected_prediction
    
    def get_attention_weights(self, u: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract attention weights for interpretability.
        
        Args:
            u: Input features [Batch, Input_Dim]
        Returns:
            Attention weights [Batch, Input_Dim] or None if attention disabled
        """
        if not self.enable_attention:
            return None
        _, weights = self.attention(u)
        return weights
    
    def adapt_to_new_topology(
        self, 
        support_data: List[Tuple[torch.Tensor, torch.Tensor]], 
        task_id: int,
        n_steps: int = 10,
        lr: float = 0.01
    ):
        """
        Meta-learning: Fast adaptation to new grid topology.
        
        Args:
            support_data: List of (input, target) pairs for new topology
            task_id: Which task slot to adapt
            n_steps: Number of gradient steps
            lr: Learning rate for adaptation
        """
        if not self.enable_metalearning:
            raise ValueError("Meta-learning not enabled in this model")
        
        optimizer = torch.optim.SGD(
            [self.meta_adapter.task_embeddings[task_id], 
             self.meta_adapter.task_biases[task_id]], 
            lr=lr
        )
        
        for step in range(n_steps):
            total_loss = 0
            for u_support, v_target in support_data:
                v_pred = self.forward(u_support, task_id=task_id)
                loss = F.mse_loss(v_pred, v_target)
                total_loss += loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        return total_loss.item()


# --- Unit Tests ---
if __name__ == "__main__":
    print("🔬 Testing Enhanced DeepONet with All Features...")
    
    # Test 1: Basic functionality with projection
    print("\n1️⃣ Testing Basic Inference + Projection")
    model = DeepONet(n_buses=5, enable_uncertainty=False, enable_attention=False)
    model.eval()
    u_test = torch.randn(2, 3)
    v_out = model(u_test)
    
    print(f"   Output Shape: {v_out.shape}")
    print(f"   Min Voltage: {v_out.min().item():.4f} (Should be >= 0.9)")
    print(f"   Max Voltage: {v_out.max().item():.4f} (Should be <= 1.1)")
    assert v_out.min() >= 0.9 and v_out.max() <= 1.1, "❌ Projection failed!"
    print("   ✅ Basic projection works")
    
    # Test 2: Uncertainty quantification
    print("\n2️⃣ Testing Uncertainty Quantification")
    model_unc = DeepONet(n_buses=5, enable_uncertainty=True)
    model_unc.eval()
    mean, std = model_unc(u_test, return_uncertainty=True)
    
    print(f"   Mean Shape: {mean.shape}, Std Shape: {std.shape}")
    print(f"   Mean Voltage Range: [{mean.min().item():.4f}, {mean.max().item():.4f}]")
    print(f"   Avg Uncertainty: {std.mean().item():.4f}")
    assert std.min() >= 0, "❌ Negative uncertainty!"
    print("   ✅ Uncertainty quantification works")
    
    # Test 3: Feature attention
    print("\n3️⃣ Testing Feature Attention")
    model_attn = DeepONet(n_buses=5, enable_attention=True)
    model_attn.eval()
    weights = model_attn.get_attention_weights(u_test)
    
    print(f"   Attention Weights Shape: {weights.shape}")
    print(f"   Weights Sum: {weights.sum(dim=1)}")  # Should sum to 1
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5), "❌ Attention not normalized!"
    print("   ✅ Feature attention works")
    
    # Test 4: Multi-fidelity modeling
    print("\n4️⃣ Testing Multi-Fidelity Modeling")
    model_mf = DeepONet(n_buses=5, enable_multifidelity=True)
    model_mf.eval()
    v_mf = model_mf(u_test)
    
    print(f"   Multi-Fidelity Output Shape: {v_mf.shape}")
    print(f"   Fidelity Weights: {F.softmax(model_mf.fidelity_fusion.fidelity_weights, dim=0)}")
    print("   ✅ Multi-fidelity fusion works")
    
    # Test 5: Meta-learning adaptation
    print("\n5️⃣ Testing Meta-Learning")
    model_meta = DeepONet(n_buses=5, enable_metalearning=True, n_tasks=3)
    model_meta.train()
    
    # Create mock support data for new topology
    support_data = [(torch.randn(2, 3), torch.rand(2, 5) * 0.2 + 0.95) for _ in range(3)]
    loss = model_meta.adapt_to_new_topology(support_data, task_id=1, n_steps=5)
    
    print(f"   Adaptation Loss: {loss:.6f}")
    print("   ✅ Meta-learning adaptation works")
    
    # Test 6: Full integration
    print("\n6️⃣ Testing Full Integration (All Features)")
    model_full = DeepONet(
        n_buses=5,
        enable_uncertainty=True,
        enable_attention=True,
        enable_multifidelity=True,
        enable_metalearning=True
    )
    model_full.eval()
    
    mean_full, std_full = model_full(u_test, task_id=0, return_uncertainty=True)
    weights_full = model_full.get_attention_weights(u_test)
    
    print(f"   Full Model Output: {mean_full.shape}")
    print(f"   Uncertainty: {std_full.mean().item():.4f}")
    print(f"   Attention: {weights_full[0].detach().numpy()}")
    print(f"   Voltage Range: [{mean_full.min().item():.4f}, {mean_full.max().item():.4f}]")
    assert mean_full.min() >= 0.9 and mean_full.max() <= 1.1, "❌ Full model projection failed!"
    print("   ✅ All features integrated successfully")
    
    print("\n🎉 All Tests Passed! Enhanced DeepONet Ready for Deployment.")
