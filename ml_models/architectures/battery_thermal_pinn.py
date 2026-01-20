"""
Battery Energy Storage System with Thermal + SOH Degradation Coupling.
Models electrochemical dynamics, thermal behavior, and aging mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class BatteryPhysicsConstraints(nn.Module):
    """
    Enforces battery physics:
    1. State of Charge (SOC): 0.1 <= SOC <= 0.9 (operational window)
    2. C-rate limits: |P| <= C_rate_max * Capacity
    3. Thermal dynamics: dT/dt = (I²R - h(T-T_amb)) / (m*c_p)
    4. SOH degradation: Calendar + Cycle aging
    """
    def __init__(self, capacity_kwh: float = 100.0, max_c_rate: float = 1.0):
        super().__init__()
        self.Q_nom = capacity_kwh * 3600  # Convert to kWh -> kJ
        self.C_rate_max = max_c_rate
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        
        # Thermal parameters
        self.R_internal = 0.01  # Internal resistance (Ohm)
        self.m = 500.0  # Mass (kg)
        self.c_p = 1000.0  # Specific heat (J/kg·K)
        self.h = 10.0  # Heat transfer coefficient (W/K)
        
        # Degradation parameters
        self.k_cal = 1e-7  # Calendar aging rate
        self.k_cyc = 1e-6  # Cycle aging rate
        
    def soc_bounds_loss(self, soc: torch.Tensor) -> torch.Tensor:
        """Soft constraint to keep SOC in operational window."""
        violation_low = torch.clamp(self.SOC_min - soc, min=0.0)
        violation_high = torch.clamp(soc - self.SOC_max, min=0.0)
        return torch.mean(violation_low ** 2 + violation_high ** 2)
    
    def c_rate_constraint(self, power_kw: torch.Tensor) -> torch.Tensor:
        """Enforce |P| <= C_rate_max * Capacity"""
        max_power = self.C_rate_max * (self.Q_nom / 3600.0)  # Back to kW
        return torch.clamp(power_kw, min=-max_power, max=max_power)
    
    def thermal_dynamics_residual(self, temp: torch.Tensor, temp_next: torch.Tensor,
                                 current_a: torch.Tensor, temp_amb: torch.Tensor, 
                                 dt: float = 1.0) -> torch.Tensor:
        """
        Physics-informed thermal equation:
        T_{t+1} = T_t + dt/mc_p * (I²R - h(T-T_amb))
        """
        heat_generation = (current_a ** 2) * self.R_internal
        heat_dissipation = self.h * (temp - temp_amb)
        
        temp_predicted = temp + (dt / (self.m * self.c_p)) * (heat_generation - heat_dissipation)
        
        return torch.mean((temp_next - temp_predicted) ** 2)
    
    def degradation_model(self, soc: torch.Tensor, temp: torch.Tensor, 
                         dod: torch.Tensor, time_hours: float) -> torch.Tensor:
        """
        State of Health (SOH) degradation:
        dSOH/dt = -k_cal * exp(E_a/RT) - k_cyc * DOD * sqrt(C_rate)
        
        Simplified for demonstration.
        """
        # Calendar aging (Arrhenius)
        E_a = 1000.0  # Activation energy (simplified)
        R = 8.314
        T_kelvin = temp + 273.15
        cal_aging = self.k_cal * torch.exp(-E_a / (R * T_kelvin)) * time_hours
        
        # Cycle aging (DOD dependent)
        cyc_aging = self.k_cyc * dod
        
        soh_loss = cal_aging + cyc_aging
        return soh_loss

class BatteryThermalPINN(nn.Module):
    """
    Physics-Informed Neural Network for Battery Management.
    
    Inputs: [SOC_t, Temp_t, Power_cmd, Temp_amb, Time_of_Day]
    Outputs: [SOC_{t+1}, Temp_{t+1}, SOH_delta, Current]
    """
    def __init__(self, input_dim: int = 5, capacity_kwh: float = 100.0):
        super().__init__()
        
        # State evolution network
        self.state_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 4)  # [SOC_next, Temp_next, SOH_delta, Current]
        )
        
        # Physics Constraints
        self.physics = BatteryPhysicsConstraints(capacity_kwh)
        
        # Uncertainty
        self.dropout = nn.Dropout(p=0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, apply_constraints: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [Batch, 5] -> [SOC_t, Temp_t, Power_cmd, Temp_amb, Time]
        Returns:
            Dictionary of predictions
        """
        output = self.state_net(self.dropout(x))
        
        soc_t = x[:, 0:1]
        temp_t = x[:, 1:2]
        power_cmd = x[:, 2:3]
        
        soc_delta = output[:, 0:1]
        temp_next = output[:, 1:2]
        soh_delta = output[:, 2:3]
        current = output[:, 3:4]
        
        if apply_constraints:
            # SOC evolution: SOC_{t+1} = SOC_t + delta
            soc_next = torch.clamp(soc_t + soc_delta, 
                                  min=self.physics.SOC_min, 
                                  max=self.physics.SOC_max)
            
            # Temperature must be positive
            temp_next = torch.clamp(temp_next, min=-10.0, max=60.0)
            
            # SOH only decreases
            soh_delta = -torch.abs(soh_delta)
            
            # Power constraint
            power_actual = self.physics.c_rate_constraint(power_cmd)
        else:
            soc_next = soc_t + soc_delta
            power_actual = power_cmd
            
        return {
            'soc_next': soc_next,
            'temp_next': temp_next,
            'soh_delta': soh_delta,
            'current': current,
            'power_actual': power_actual
        }
    
    def physics_loss(self, x: torch.Tensor, predictions: Dict[str, torch.Tensor], 
                    dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Computes physics-informed loss components.
        """
        soc_t = x[:, 0:1]
        temp_t = x[:, 1:2]
        power_cmd = x[:, 2:3]
        temp_amb = x[:, 3:4]
        
        soc_next = predictions['soc_next']
        temp_next = predictions['temp_next']
        soh_delta = predictions['soh_delta']
        current = predictions['current']
        
        # Loss 1: SOC bounds
        loss_soc_bounds = self.physics.soc_bounds_loss(soc_next)
        
        # Loss 2: Coulomb counting (SOC = SOC_0 + integral(I*dt/Q))
        # Assuming voltage ~constant, P = V*I => I = P/V
        V_nom = 400.0  # Nominal voltage (V)
        current_from_power = power_cmd / V_nom
        soc_expected = soc_t + (current_from_power * dt) / (self.physics.Q_nom / 3600.0)
        loss_coulomb = torch.mean((soc_next - soc_expected) ** 2)
        
        # Loss 3: Thermal dynamics
        loss_thermal = self.physics.thermal_dynamics_residual(
            temp_t, temp_next, current, temp_amb, dt
        )
        
        # Loss 4: Degradation consistency
        dod = torch.abs(soc_next - soc_t)  # Depth of Discharge
        soh_expected = self.physics.degradation_model(soc_t, temp_t, dod, dt)
        loss_degradation = torch.mean((torch.abs(soh_delta) - soh_expected) ** 2)
        
        # Loss 5: Power-Current consistency
        loss_power = torch.mean((current * V_nom - power_cmd) ** 2)
        
        return {
            'soc_bounds': loss_soc_bounds,
            'coulomb_counting': loss_coulomb,
            'thermal_dynamics': loss_thermal,
            'degradation': loss_degradation,
            'power_consistency': loss_power
        }
    
    def uncertainty_estimate(self, x: torch.Tensor, n_samples: int = 20) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Monte Carlo Dropout for uncertainty."""
        self.train()
        
        soc_samples = []
        temp_samples = []
        soh_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, apply_constraints=True)
                soc_samples.append(pred['soc_next'])
                temp_samples.append(pred['temp_next'])
                soh_samples.append(pred['soh_delta'])
        
        self.eval()
        
        return {
            'soc': (torch.stack(soc_samples).mean(0), torch.stack(soc_samples).std(0)),
            'temp': (torch.stack(temp_samples).mean(0), torch.stack(temp_samples).std(0)),
            'soh': (torch.stack(soh_samples).mean(0), torch.stack(soh_samples).std(0))
        }
    
    def explain_prediction(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Gradient-based feature importance."""
        x.requires_grad_(True)
        pred = self.forward(x, apply_constraints=False)
        
        # Compute importance for SOC prediction
        pred['soc_next'].sum().backward(retain_graph=True)
        importance_soc = torch.abs(x.grad * x)
        
        x.grad.zero_()
        pred['temp_next'].sum().backward()
        importance_temp = torch.abs(x.grad * x)
        
        feature_names = ['SOC_t', 'Temp_t', 'Power_cmd', 'Temp_amb', 'Time']
        
        result = {
            'SOC': {name: importance_soc[:, i].mean().item() 
                   for i, name in enumerate(feature_names)},
            'Temp': {name: importance_temp[:, i].mean().item() 
                    for i, name in enumerate(feature_names)}
        }
        
        x.requires_grad_(False)
        return result

# --- Unit Test ---
if __name__ == "__main__":
    batch_size = 5
    
    # Test scenarios
    soc_t = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6]).unsqueeze(1)
    temp_t = torch.tensor([25.0, 30.0, 20.0, 35.0, 22.0]).unsqueeze(1)
    power_cmd = torch.tensor([50.0, -30.0, 20.0, -40.0, 0.0]).unsqueeze(1)  # kW
    temp_amb = torch.tensor([20.0, 25.0, 15.0, 30.0, 18.0]).unsqueeze(1)
    time = torch.tensor([12.0, 13.0, 14.0, 15.0, 16.0]).unsqueeze(1)
    
    x = torch.cat([soc_t, temp_t, power_cmd, temp_amb, time], dim=-1)
    
    model = BatteryThermalPINN(input_dim=5, capacity_kwh=100.0)
    
    # Forward pass
    predictions = model(x)
    
    print("=" * 50)
    print("BATTERY THERMAL PINN TEST")
    print("=" * 50)
    
    for i in range(batch_size):
        print(f"\nScenario {i+1}:")
        print(f"  SOC: {soc_t[i].item():.2f} -> {predictions['soc_next'][i].item():.2f}")
        print(f"  Temp: {temp_t[i].item():.1f}°C -> {predictions['temp_next'][i].item():.1f}°C")
        print(f"  SOH Delta: {predictions['soh_delta'][i].item():.6f}")
        print(f"  Current: {predictions['current'][i].item():.2f} A")
    
    # Physics loss
    physics_losses = model.physics_loss(x, predictions, dt=1.0)
    print("\nPhysics Losses:")
    for key, val in physics_losses.items():
        print(f"  {key}: {val.item():.6f}")
    
    # Uncertainty
    uncertainty = model.uncertainty_estimate(x[:2], n_samples=10)
    print("\nUncertainty (first sample):")
    for key, (mean, std) in uncertainty.items():
        print(f"  {key}: {mean[0].item():.4f} ± {std[0].item():.4f}")
    
    # Explainability
    importance = model.explain_prediction(x[:2])
    print("\nFeature Importance for SOC:")
    for feature, score in sorted(importance['SOC'].items(), key=lambda x: -x[1]):
        print(f"  {feature}: {score:.4f}")
