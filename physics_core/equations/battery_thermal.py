"""
Battery Thermal Equation Module.
Implements the Lumped Capacitance Thermal Model for BESS (Battery Energy Storage Systems).
Critical for enforcing safety constraints and preventing thermal runaway in the Digital Twin.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class BatteryThermalEquation(nn.Module):
    """
    Physics-Informed implementation of Battery Thermodynamics.
    
    Principles:
    - First Principle: Conservation of Energy (Thermal).
    - Differentiable: Supports Autograd for PINN training.
    - Safety-Critical: Models the physical limits of charge/discharge rates.
    
    Governing Equation:
    m * Cp * (dT/dt) = (I^2 * R) - (h * A * (T - T_amb))
    [Heat Stored]    = [Heat Gen] - [Heat Dissipated]
    """
    
    def __init__(self, 
                 mass_kg: float, 
                 specific_heat_cp: float, 
                 internal_resistance_r: float, 
                 heat_transfer_coeff_h: float, 
                 surface_area_a: float):
        """
        Args:
            mass_kg (m): Mass of the battery cell/pack (kg).
            specific_heat_cp (Cp): Specific heat capacity (J/kg.K).
            internal_resistance_r (R): Average internal impedance (Ohms).
            heat_transfer_coeff_h (h): Convective cooling coefficient (W/m^2.K).
            surface_area_a (A): Effective cooling surface area (m^2).
        """
        super().__init__()
        
        # 1. Fail Fast: Validate physical constants
        if mass_kg <= 0 or specific_heat_cp <= 0:
            raise ValueError("Physics Violation: Mass and Specific Heat must be positive.")
        if internal_resistance_r < 0:
            raise ValueError("Physics Violation: Resistance cannot be negative (Active generation violation).")

        # 2. Register constants as buffers (Part of model state, but non-trainable)
        self.register_buffer("m_cp", torch.tensor(mass_kg * specific_heat_cp, dtype=torch.float32))
        self.register_buffer("r_int", torch.tensor(internal_resistance_r, dtype=torch.float32))
        self.register_buffer("h_a", torch.tensor(heat_transfer_coeff_h * surface_area_a, dtype=torch.float32))

    def forward(self, 
                temp_c: torch.Tensor, 
                d_temp_dt: torch.Tensor, 
                current_a: torch.Tensor, 
                temp_amb_c: torch.Tensor,
                dynamic_resistance: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the Thermal Residual (The violation of thermodynamic laws).
        
        Args:
            temp_c (T): Battery Temperature (Celsius) [Batch, 1]
            d_temp_dt (dT/dt): Rate of change of temperature (Celsius/sec) [Batch, 1]
            current_a (I): Charge/Discharge Current (Amps) [Batch, 1]
            temp_amb_c (T_amb): Ambient Environment Temperature (Celsius) [Batch, 1]
            dynamic_resistance (R_dyn): Optional tensor for SOC/Temp-dependent resistance.
                                        If provided, overrides static r_int.
            
        Returns:
            residual: The imbalance in the heat equation (should be 0).
        """
        # --- Term 1: Heat Generation (Joule Heating) ---
        # Q_gen = I^2 * R
        # Use dynamic resistance map if available (High Fidelity), else static (Low Fidelity)
        r_eff = dynamic_resistance if dynamic_resistance is not None else self.r_int
        q_gen = (current_a ** 2) * r_eff
        
        # --- Term 2: Heat Dissipation (Newton's Law of Cooling) ---
        # Q_diss = h * A * (T_cell - T_amb)
        q_diss = self.h_a * (temp_c - temp_amb_c)
        
        # --- Term 3: Thermal Inertia (Heat Storage) ---
        # Q_stored = m * Cp * dT/dt
        q_stored = self.m_cp * d_temp_dt
        
        # --- The Residual ---
        # Law: Rate of Storage = Generation - Dissipation
        # Residual = Storage - (Gen - Diss)
        residual = q_stored - (q_gen - q_diss)
        
        return residual

    def __repr__(self):
        return (f"BatteryThermalEquation("
                f"Thermal Mass={self.m_cp.item():.1f} J/K, "
                f"R_int={self.r_int.item()*1000:.1f} mΩ)")
                   
