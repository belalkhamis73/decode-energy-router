"""
Swing Equation Module.
Defines the differential algebraic equations (DAEs) governing rotor angle stability.
This is the "Ground Truth" that the PINN must satisfy during training.
"""

import torch
import torch.nn as nn
from typing import Tuple

class SwingEquation(nn.Module):
    """
    Physics-Informed implementation of the Classical Swing Equation.
    
    Principles:
    - OOP: Encapsulates physical constants (Inertia, Damping) within the object.
    - Differentiable: Inherits from nn.Module to support PyTorch Autograd.
    - Fail Fast: Validates physical feasibility of parameters on initialization.
    
    Governing System:
    1. d(delta)/dt = omega * omega_s
    2. 2H * d(omega)/dt = Pm - Pe - D * omega
    """
    
    def __init__(self, inertia_h: float, damping_d: float, sync_freq_hz: float = 60.0):
        """
        Args:
            inertia_h (H): Inertia constant (seconds). Represents the stored energy.
            damping_d (D): Damping coefficient (p.u.). Represents friction/losses.
            sync_freq_hz (fs): Synchronous grid frequency (usually 50Hz or 60Hz).
        """
        super().__init__()
        
        # 1. Fail Fast: Prevent unphysical simulation parameters
        if inertia_h <= 0:
            raise ValueError(f"Physics Violation: Inertia (H) must be positive. Got {inertia_h}")
        if damping_d < 0:
            raise ValueError(f"Physics Violation: Damping (D) cannot be negative. Got {damping_d}")

        # 2. Tensor Registration:
        # Register as buffers so they are part of the state_dict (saved with model)
        # but are NOT treated as trainable parameters (gradients frozen).
        self.register_buffer("two_h", torch.tensor(2.0 * inertia_h, dtype=torch.float32))
        self.register_buffer("d_dump", torch.tensor(damping_d, dtype=torch.float32))
        self.register_buffer("omega_s", torch.tensor(2.0 * 3.14159 * sync_freq_hz, dtype=torch.float32))

    def forward(self, 
                omega: torch.Tensor, 
                d_delta_dt: torch.Tensor, 
                d_omega_dt: torch.Tensor, 
                p_mech: torch.Tensor, 
                p_elec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the Physics Residual (The Error in the Law).
        Ideally, these return values should be Zero.
        
        Args:
            omega: Angular frequency deviation in p.u. (Network Output)
            d_delta_dt: Time derivative of Rotor Angle (computed via Autograd)
            d_omega_dt: Time derivative of Frequency (computed via Autograd)
            p_mech: Mechanical Power Input (Control Variable)
            p_elec: Electrical Power Output (Grid State)
            
        Returns:
            (res_delta, res_omega): The violation of Angle and Frequency laws.
        """
        
        # --- Law 1: Kinematic Relationship ---
        # The change in angle is proportional to the speed deviation.
        # d(delta)/dt = omega * omega_s
        res_delta = d_delta_dt - (omega * self.omega_s)
        
        # --- Law 2: Newton's Second Law for Rotation ---
        # Inertia * Acceleration = Accelerating Torque - Damping
        # 2H * d(omega)/dt = (Pm - Pe) - D * omega
        
        # Calculate forces
        accelerating_power = p_mech - p_elec
        damping_force = self.d_dump * omega
        inertial_force = self.two_h * d_omega_dt
        
        # Calculate residual (Validation of Energy Conservation)
        res_omega = inertial_force - (accelerating_power - damping_force)
        
        return res_delta, res_omega

    def __repr__(self):
        """Self-documenting string representation."""
        return f"SwingEquation(H={self.two_h.item()/2:.2f}s, D={self.d_dump.item():.2f}pu)"

# --- CRITICAL ADDITION: The Helper Function ---
def calculate_freq_deviation(power_mismatch: float, inertia_h: float) -> float:
    """
    Simplified frequency deviation calculation.
    df/dt = (Pm - Pe) / (2H)
    """
    return power_mismatch / (2 * inertia_h)

def calculate_freq_deviation(pm, h): return pm / (2*h) if h>0 else 0.0
