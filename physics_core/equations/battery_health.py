"""
Battery Health (SOH) Physics Module.
Implements degradation dynamics based on Arrhenius (Calendar) and Throughput (Cycle) aging.

References:
- "Predicting the life of Li-ion batteries using the Arrhenius model"
- "Assessing battery degradation as a key performance indicator"
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class BatteryParams:
    """Calibrated parameters for Li-Ion NMC chemistry."""
    cycle_k: float = 3e-5       # Cycle aging coefficient (per Ah)
    cal_k: float = 1.5e-5       # Calendar aging coefficient (per hour at ref temp)
    ea: float = 22406.0         # Activation Energy (J/mol)
    gas_constant: float = 8.314 # J/(mol.K)
    ref_temp_k: float = 298.15  # 25 Celsius in Kelvin

class BatteryHealthEquation(nn.Module):
    def __init__(self, params: BatteryParams = BatteryParams()):
        super().__init__()
        self.p = params

    def forward(self, 
                current_soh: float, 
                temp_c: float, 
                current_a: float, 
                dt_hours: float = 1.0) -> float:
        """
        Calculates the new State of Health (SOH) after a time step.
        
        Args:
            current_soh: Current SOH (0.0 to 1.0).
            temp_c: Cell Temperature (Celsius) from Thermal Model.
            current_a: Current flow (Amps). Abs value used for throughput.
            dt_hours: Time step duration.
            
        Returns:
            new_soh: Updated SOH.
        """
        # 1. Physics: Cycle Aging (Throughput)
        # Loss is proportional to total charge moved (Ah)
        throughput_ah = abs(current_a) * dt_hours
        loss_cycle = self.p.cycle_k * throughput_ah

        # 2. Physics: Calendar Aging (Arrhenius Equation)
        # Loss accelerates exponentially with temperature
        # k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
        temp_k = temp_c + 273.15
        exponent = (self.p.ea / self.p.gas_constant) * (1/self.p.ref_temp_k - 1/temp_k)
        # Clamp exponent to prevent overflow in extreme synthetic scenarios
        exponent = np.clip(exponent, -50, 50) 
        
        accel_factor = np.exp(exponent)
        loss_calendar = self.p.cal_k * dt_hours * accel_factor

        # 3. State Update
        total_degradation = loss_cycle + loss_calendar
        new_soh = current_soh - total_degradation
        
        return max(0.0, min(1.0, new_soh))

# --- Unit Test ---
if __name__ == "__main__":
    model = BatteryHealthEquation()
    
    # Scenario: Hot day (45C), Discharging hard (50A)
    soh_new = model.forward(current_soh=1.0, temp_c=45.0, current_a=50.0)
    
    print(f"Initial SOH: 1.0")
    print(f"New SOH:     {soh_new:.6f}")
    print(f"Degradation: {(1.0 - soh_new)*100:.4f}%")# physics_core/equations/battery_health.py
