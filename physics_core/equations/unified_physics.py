# NEW FILE: physics_core/equations/unified_physics.py
"""
Unified Physics Validation Engine
Combines all governing equations into a single differentiable loss function
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .power_flow import PowerFlowEquation
from .swing_equation import SwingEquation, calculate_freq_deviation
from .battery_thermal import BatteryThermalEquation
from .battery_health import BatteryHealthEquation

class UnifiedPhysicsValidator(nn.Module):
    """
    The "Ground Truth" Engine for the Digital Twin.
    Aggregates all physics residuals into a composite loss.
    """
    def __init__(self, grid_topology: Dict):
        super().__init__()
        
        # Initialize sub-modules
        self.power_flow = PowerFlowEquation(grid_topology['n_buses'])
        self.swing_eq = SwingEquation(
            inertia_h=grid_topology.get('inertia_h', 3.5),
            damping_d=grid_topology.get('damping_d', 1.0)
        )
        self.thermal = BatteryThermalEquation(
            mass_kg=250.0, specific_heat_cp=900.0,
            internal_resistance_r=0.05, heat_transfer_coeff_h=15.0,
            surface_area_a=2.5
        )
        
        # Loss weights (configurable via config.py)
        self.register_buffer('w_power', torch.tensor(1.0))
        self.register_buffer('w_swing', torch.tensor(0.5))
        self.register_buffer('w_thermal', torch.tensor(0.3))
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                measurements: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate physics residuals across all domains.
        
        Returns:
            residuals: Dictionary of individual physics violations
            composite_loss: Weighted sum for backpropagation
        """
        residuals = {}
        
        # 1. Power Flow Residuals (KCL/KVL)
        res_P, res_Q = self.power_flow(
            predictions['V_mag'], predictions['V_ang'],
            measurements['P_injection'], measurements['Q_injection']
        )
        residuals['power_balance'] = torch.norm(res_P) + torch.norm(res_Q)
        
        # 2. Swing Equation Residuals (Frequency Dynamics)
        res_delta, res_omega = self.swing_eq(
            predictions['omega'], predictions['d_delta_dt'], 
            predictions['d_omega_dt'],
            measurements['P_mech'], measurements['P_elec']
        )
        residuals['swing_violation'] = torch.norm(res_delta) + torch.norm(res_omega)
        
        # 3. Thermal Residuals (Battery Safety)
        if 'battery_temp' in predictions:
            res_thermal = self.thermal(
                predictions['battery_temp'], predictions['d_temp_dt'],
                measurements['battery_current'], measurements['temp_ambient']
            )
            residuals['thermal_violation'] = torch.norm(res_thermal)
        
        # Composite Loss
        composite = (self.w_power * residuals['power_balance'] +
                     self.w_swing * residuals['swing_violation'])
        
        if 'thermal_violation' in residuals:
            composite += self.w_thermal * residuals['thermal_violation']
        
        return {
            'residuals': residuals,
            'composite_loss': composite,
            'is_valid': composite < 1e-4  # Physics compliance threshold
                  }
