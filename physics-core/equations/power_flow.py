"""
Power Flow Equation Module.
Implements the AC Power Flow (Kirchhoff's Laws) as differentiable residuals.
Crucial for enforcing energy conservation in the Physics-Informed Neural Network.
"""

import torch
import torch.nn as nn
from typing import Tuple

class PowerFlowEquation(nn.Module):
    """
    Physics-Informed implementation of AC Power Flow Equations.
    
    Principles:
    - Vectorized Operations: Handles computation for N buses simultaneously.
    - Differentiable: Supports PyTorch Autograd for PINN training.
    - Statutory Compliance: Enforces P and Q balance at every node.
    
    Governing Equations:
    P_i = Sum_j |V_i||V_j|(G_ij cos(theta_ij) + B_ij sin(theta_ij))
    Q_i = Sum_j |V_i||V_j|(G_ij sin(theta_ij) - B_ij cos(theta_ij))
    """
    
    def __init__(self, num_buses: int):
        super().__init__()
        self.num_buses = num_buses
        
        # Admittance Matrix (Y_bus) components: G (Conductance) and B (Susceptance)
        # Registered as buffers to be saved with the model but not trained
        self.register_buffer("G_bus", torch.zeros(num_buses, num_buses))
        self.register_buffer("B_bus", torch.zeros(num_buses, num_buses))

    def set_admittance_matrix(self, Y_bus: torch.Tensor):
        """
        Populates the grid topology from a complex adjacency matrix.
        Args:
            Y_bus: Complex tensor of shape (N, N) representing grid admittance.
        """
        if Y_bus.shape != (self.num_buses, self.num_buses):
            raise ValueError(f"Topology Mismatch: Expected {self.num_buses}x{self.num_buses}, got {Y_bus.shape}")
            
        self.G_bus = Y_bus.real.float()
        self.B_bus = Y_bus.imag.float()

    def forward(self, 
                V_mag: torch.Tensor, 
                V_ang: torch.Tensor, 
                P_in: torch.Tensor, 
                Q_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Power Flow Residuals (The "Energy Conservation Violation").
        
        Args:
            V_mag: Voltage Magnitudes [Batch, N]
            V_ang: Voltage Angles (Radians) [Batch, N]
            P_in: Net Active Power Injection (Gen - Load) [Batch, N]
            Q_in: Net Reactive Power Injection (Gen - Load) [Batch, N]
            
        Returns:
            (res_P, res_Q): Residuals for Active and Reactive power balance.
        """
        batch_size = V_mag.shape[0]
        
        # 1. Compute Phase Angle Differences (Theta_ij = Theta_i - Theta_j)
        # Broadcasting to create an (N, N) matrix for each sample in batch
        # Result shape: [Batch, N, N]
        theta_ij = V_ang.unsqueeze(2) - V_ang.unsqueeze(1)
        
        # 2. Compute Trigonometric Terms
        cos_theta = torch.cos(theta_ij)
        sin_theta = torch.sin(theta_ij)
        
        # 3. Compute Power Flow Terms (The "Physics" Calculation)
        # Term: |V_j| * (G_ij * cos + B_ij * sin)
        # We need to broadcast G_bus/B_bus to match batch size
        G = self.G_bus.unsqueeze(0).expand(batch_size, -1, -1)
        B = self.B_bus.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Inner summation term for P_i calculation
        term_p = V_mag.unsqueeze(1) * (G * cos_theta + B * sin_theta)
        
        # Inner summation term for Q_i calculation
        term_q = V_mag.unsqueeze(1) * (G * sin_theta - B * cos_theta)
        
        # 4. Calculate Calculated Power (P_calc, Q_calc)
        # Sum over index j (columns)
        P_calc = V_mag * torch.sum(term_p, dim=2)
        Q_calc = V_mag * torch.sum(term_q, dim=2)
        
        # 5. Compute Residuals (Violation = Calculated - Injected)
        res_P = P_calc - P_in
        res_Q = Q_calc - Q_in
        
        return res_P, res_Q

    def __repr__(self):
        return f"PowerFlowEquation(buses={self.num_buses})"
                
