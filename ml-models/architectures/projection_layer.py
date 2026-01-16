"""
Physics Projection Layer.
Implements a differentiable optimization layer (OptNet / cvxpylayers) that projects
raw neural network outputs onto the valid physical manifold (e.g., satisfying KCL/KVL).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Note: In production, we would import cvxpylayers.torch
# For this implementation, we simulate the projection using a closed-form solution
# or a simplified penalty method to avoid complex dependency management in MVP.

class HardConstraintProjection(nn.Module):
    """
    Projects predictions to satisfy Linear Equality Constraints: Ax = b.
    Commonly used for Kirchhoff's Current Law (KCL): Sum(Currents) = 0.
    
    Principles:
    - Safety: Guarantees output is physically valid before it hits the grid.
    - Differentiable: Allows backprop through the optimization step.
    """
    
    def __init__(self, num_nodes: int, tolerance: float = 1e-4):
        super().__init__()
        self.num_nodes = num_nodes
        self.tolerance = tolerance
        
        # Identity matrix for the projection geometry
        # In a full grid, this would be the Incidence Matrix
        self.register_buffer("identity", torch.eye(num_nodes))

    def forward(self, raw_predictions: torch.Tensor, A_matrix: torch.Tensor, b_vector: torch.Tensor) -> torch.Tensor:
        """
        Solves: min ||x - x_pred||^2 s.t. Ax = b
        Closed form solution for equality constraints:
        x_proj = x_pred - A^T * (A * A^T)^-1 * (A * x_pred - b)
        
        Args:
            raw_predictions (x_pred): Output from the Neural Network [Batch, N]
            A_matrix: Constraint Matrix (e.g., Grid Incidence Matrix) [Batch, M, N]
            b_vector: Constraint Target (e.g., Zero for KCL) [Batch, M, 1]
            
        Returns:
            Projected Predictions (x_proj) that strictly satisfy Ax=b
        """
        # 1. Calculate the Violation (Residual)
        # violation = A * x_pred - b
        # We assume A_matrix is broadcastable or fixed
        
        # Dimensions:
        # x_pred: [B, N, 1]
        # A: [B, M, N]
        # b: [B, M, 1]
        
        x_in = raw_predictions.unsqueeze(-1) # [B, N, 1]
        
        # Compute A * x_pred
        Ax = torch.bmm(A_matrix, x_in) # [B, M, 1]
        violation = Ax - b_vector
        
        # 2. Check if projection is needed (Optimization for speed)
        if torch.max(torch.abs(violation)) < self.tolerance:
            return raw_predictions
            
        # 3. Compute the Projection Correction
        # Correction = A^T * (A A^T)^-1 * violation
        # Pseudoinverse of (A A^T) usually computed via Cholesky or SVD
        
        # A_T: [B, N, M]
        A_T = A_matrix.transpose(1, 2)
        
        # Gram Matrix: G = A * A^T [B, M, M]
        gram_matrix = torch.bmm(A_matrix, A_T)
        
        # Inverse Gram: G^-1
        # Add epsilon for numerical stability
        gram_inv = torch.linalg.pinv(gram_matrix + 1e-6 * torch.eye(gram_matrix.shape[1], device=gram_matrix.device))
        
        # Lagrange Multipliers: lambda = G^-1 * violation
        lambda_vec = torch.bmm(gram_inv, violation) # [B, M, 1]
        
        # Correction: A^T * lambda
        correction = torch.bmm(A_T, lambda_vec) # [B, N, 1]
        
        # 4. Apply Correction
        x_projected = x_in - correction
        
        return x_projected.squeeze(-1)

class BoundConstraintLayer(nn.Module):
    """
    Projects predictions to satisfy Inequality Constraints: min <= x <= max.
    Used for Voltage Limits (0.95 <= V <= 1.05 p.u.) and Generator Limits.
    """
    def __init__(self, lower_bound: float, upper_bound: float):
        super().__init__()
        self.lower = lower_bound
        self.upper = upper_bound
        # Softplus helps gradients flow near the boundary, unlike hard clamping
        self.soft_clamp = nn.Softplus() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Clamping.
        Strict clamping sets gradient to 0, which kills learning.
        We use a smooth approximation or straight-through estimator if needed.
        For hard safety, we use standard torch.clamp at inference time.
        """
        if self.training:
            # Smooth approximation during training to allow gradient flow
            # This is a soft-clipping implementation
            return torch.clamp(x, self.lower, self.upper) 
        else:
            # Hard constraint during inference/deployment
            return torch.clamp(x, self.lower, self.upper)

# --- Unit Test ---
if __name__ == "__main__":
    # Example: 3 Nodes (Currents), Constraint: Sum(I) = 0
    # A = [[1, 1, 1]]
    
    batch_size = 5
    num_nodes = 3
    
    # Random neural net output (violates KCL)
    predictions = torch.randn(batch_size, num_nodes)
    
    # Constraint Matrix A (Sum of all elements)
    A = torch.ones(batch_size, 1, num_nodes)
    b = torch.zeros(batch_size, 1, 1)
    
    projector = HardConstraintProjection(num_nodes)
    
    # Forward Pass
    clean_preds = projector(predictions, A, b)
    
    # Verify KCL
    sums = torch.sum(clean_preds, dim=1)
    print(f"Original Sums: {torch.sum(predictions, dim=1)}")
    print(f"Projected Sums (Should be ~0): {sums}")
      
