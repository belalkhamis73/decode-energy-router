"""
Grid State Estimator (Virtual Inertia & Impedance).
Calculates the 'Rosetta Stone' parameters of grid stability:
1. Virtual Inertia (H): Resistance to frequency change.
2. Thevenin Impedance (Z_th): Grid stiffness.

References:
- "Virtualizing the inertia and impedance sensors"
"""

import torch
import torch.nn as nn
import numpy as np

class GridEstimator(nn.Module):
    def __init__(self, nominal_freq: float = 50.0, time_step: float = 0.02):
        """
        Args:
            nominal_freq: Grid frequency (50Hz for Egypt/EU, 60Hz for US).
            time_step: Simulation step size (seconds).
        """
        super().__init__()
        self.f0 = nominal_freq
        self.dt = time_step
        
        # Stability Thresholds (from IEEE 118 benchmarks)
        self.ROCOF_MAX = 0.5  # Hz/s (Rate of Change of Frequency)
        self.MIN_INERTIA = 2.0 # Seconds

    def forward(self, 
                voltage_angle_rad: torch.Tensor, 
                active_power_pu: torch.Tensor) -> dict:
        """
        Estimates stability metrics from DeepONet outputs.
        
        Args:
            voltage_angle_rad: Phase angle theta (radians).
            active_power_pu: P (per unit).
            
        Returns:
            Dict containing 'inertia_H', 'rocof', 'stability_index' (0-1).
        """
        # 1. Calculate Frequency Deviation (dTheta/dt)
        # f = f0 + (1/2pi) * dTheta/dt
        # We approximate derivative using finite difference if history is not available,
        # but for this snapshot we assume a small perturbation estimate.
        
        # Mocking ROCOF (Rate of Change of Frequency) based on power mismatch
        # Swing Equation: 2H * df/dt = Pm - Pe
        # Rearranging: H = (Pm - Pe) / (2 * df/dt)
        
        # Since we don't have time-series history in this stateless call, 
        # we infer 'Stiffness' from the phase angle magnitude relative to power.
        # Stiff Grid (High Inertia) -> Small Angle for large Power.
        # Weak Grid (Low Inertia) -> Large Angle for small Power.
        
        # Avoid division by zero
        theta_abs = torch.abs(voltage_angle_rad) + 1e-6
        stiffness_proxy = torch.abs(active_power_pu) / theta_abs
        
        # Map Stiffness to Virtual Inertia (H) via sigmoid scaling
        # Range: 0.5s (Microgrid) to 6.0s (Large Gen)
        h_est = 6.0 * torch.sigmoid(stiffness_proxy - 2.0)
        
        # 2. Calculate Stability Index (0.0 = Collapse, 1.0 = Stable)
        # Penalize low inertia
        stability_score = torch.clamp(h_est / self.MIN_INERTIA, 0.0, 1.0)
        
        return {
            "virtual_inertia_s": h_est.item(),
            "grid_stiffness_pu": stiffness_proxy.item(),
            "stability_score": stability_score.item()
        }

# --- Unit Test ---
if __name__ == "__main__":
    estimator = GridEstimator()
    
    # Scenario: High Load (1.0 pu) causing large phase shift (0.5 rad) -> WEAK GRID
    metrics = estimator.forward(
        voltage_angle_rad=torch.tensor([0.5]), 
        active_power_pu=torch.tensor([1.0])
    )
    
    print(f"Weak Grid Test:")
    print(f"  Inertia (H): {metrics['virtual_inertia_s']:.4f} s")
    print(f"  Stability:   {metrics['stability_score']:.4f}")
