"""
Wind Turbine Model with Betz Limit Enforcement.
Implements physics-informed wind power prediction with aerodynamic constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class WindPhysicsConstraints(nn.Module):
    """
    Enforces wind turbine physics:
    1. Betz Limit: C_p <= 16/27 ≈ 0.593 (theoretical maximum efficiency)
    2. Power Curve: P = 0.5 * rho * A * v^3 * C_p
    3. Cut-in, Rated, Cut-out wind speeds
    """
    def __init__(self, rated_power_kw: float = 2000.0, rotor_diameter_m: float = 80.0):
        super().__init__()
        self.P_rated = rated_power_kw
        self.D = rotor_diameter_m
        self.A = np.pi * (rotor_diameter_m / 2) ** 2  # Swept area
        
        # Betz limit
        self.C_p_max = 16.0 / 27.0  # 0.593
        
        # Operational wind speed thresholds
        self.v_cut_in = 3.0  # m/s
        self.v_rated = 12.0  # m/s
        self.v_cut_out = 25.0  # m/s
        
    def theoretical_power(self, wind_speed: torch.Tensor, air_density: torch.Tensor, 
                         C_p: torch.Tensor) -> torch.Tensor:
        """
        Fundamental wind power equation:
        P = 0.5 * rho * A * v^3 * C_p
        """
        power = 0.5 * air_density * self.A * (wind_speed ** 3) * C_p / 1000.0  # kW
        return power
    
    def air_density(self, temp_celsius: torch.Tensor, pressure_pa: torch.Tensor) -> torch.Tensor:
        """
        Ideal gas law: rho = P / (R * T)
        Where R = 287.05 J/(kg·K) for dry air
        """
        R = 287.05
        T_kelvin = temp_celsius + 273.15
        rho = pressure_pa / (R * T_kelvin)
        return torch.clamp(rho, min=0.9, max=1.4)  # Physical bounds for sea level
    
    def enforce_betz_limit(self, C_p: torch.Tensor) -> torch.Tensor:
        """Project efficiency coefficient to valid range: 0 <= C_p <= 0.593"""
        return torch.clamp(C_p, min=0.0, max=self.C_p_max)
    
    def operational_constraints(self, wind_speed: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        """
        Apply cut-in, rated, and cut-out wind speed logic.
        """
        # Below cut-in: No power
        mask_below = (wind_speed < self.v_cut_in).float()
        
        # Above cut-out: No power (turbine shuts down for safety)
        mask_above = (wind_speed > self.v_cut_out).float()
        
        # Between rated and cut-out: Constant rated power
        mask_rated = ((wind_speed >= self.v_rated) & (wind_speed <= self.v_cut_out)).float()
        
        power_constrained = power * (1 - mask_below) * (1 - mask_above)
        power_constrained = torch.where(
            mask_rated.bool(),
            torch.clamp(power_constrained, max=self.P_rated),
            power_constrained
        )
        
        return power_constrained

class WindPINN(nn.Module):
    """
    Physics-Informed Neural Network for Wind Turbine Power.
    
    Inputs: [Wind_Speed, Wind_Direction, Temperature, Pressure, Turbulence_Intensity]
    Outputs: [Power_kW, C_p_coefficient]
    """
    def __init__(self, input_dim: int = 5, rated_power_kw: float = 2000.0, 
                 rotor_diameter_m: float = 80.0):
        super().__init__()
        
        # Neural Network Backbone
        self.power_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2)  # Outputs: [Power, C_p]
        )
        
        # Physics Constraints
        self.physics = WindPhysicsConstraints(rated_power_kw, rotor_diameter_m)
        
        # Uncertainty estimation
        self.dropout = nn.Dropout(p=0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, apply_constraints: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, 5] -> [Wind_Speed, Direction, Temp, Pressure, Turbulence]
        Returns:
            power_pred: Predicted power output (kW)
            C_p_pred: Predicted power coefficient
        """
        output = self.power_net(self.dropout(x))
        power_raw = output[:, 0:1]
        C_p_raw = output[:, 1:2]
        
        if apply_constraints:
            # Enforce Betz limit
            C_p_pred = self.physics.enforce_betz_limit(torch.sigmoid(C_p_raw) * 0.65)
            
            # Apply operational constraints
            wind_speed = x[:, 0:1]
            power_pred = self.physics.operational_constraints(wind_speed, power_raw)
            power_pred = torch.clamp(power_pred, min=0.0)
        else:
            C_p_pred = C_p_raw
            power_pred = power_raw
            
        return power_pred, C_p_pred
    
    def physics_loss(self, x: torch.Tensor, power_pred: torch.Tensor, 
                    C_p_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes physics-informed loss components.
        """
        wind_speed = x[:, 0:1]
        temp = x[:, 2:3]
        pressure = x[:, 3:4]
        
        # Compute air density from atmospheric conditions
        rho = self.physics.air_density(temp, pressure)
        
        # Loss 1: Theoretical Power Equation Consistency
        power_theoretical = self.physics.theoretical_power(wind_speed, rho, C_p_pred)
        loss_power_equation = torch.mean((power_pred - power_theoretical) ** 2)
        
        # Loss 2: Betz Limit Violation Penalty
        betz_violation = torch.clamp(C_p_pred - self.physics.C_p_max, min=0.0)
        loss_betz = torch.mean(betz_violation ** 2)
        
        # Loss 3: Cut-in Constraint (v < v_cut_in => P = 0)
        mask_cut_in = (wind_speed < self.physics.v_cut_in).float()
        loss_cut_in = torch.mean(mask_cut_in * power_pred ** 2)
        
        # Loss 4: Cut-out Constraint (v > v_cut_out => P = 0)
        mask_cut_out = (wind_speed > self.physics.v_cut_out).float()
        loss_cut_out = torch.mean(mask_cut_out * power_pred ** 2)
        
        # Loss 5: Monotonicity (power should generally increase with wind speed in region II)
        # We check that dP/dv > 0 for v in [v_cut_in, v_rated]
        region_ii_mask = ((wind_speed >= self.physics.v_cut_in) & 
                          (wind_speed <= self.physics.v_rated)).float()
        
        # Approximate derivative using finite differences
        if wind_speed.requires_grad:
            grad_power = torch.autograd.grad(
                outputs=power_pred, 
                inputs=wind_speed,
                grad_outputs=torch.ones_like(power_pred),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Penalize negative gradients in region II
            loss_monotonicity = torch.mean(region_ii_mask * torch.clamp(-grad_power, min=0.0) ** 2)
        else:
            loss_monotonicity = torch.tensor(0.0)
        
        return {
            'power_equation': loss_power_equation,
            'betz_limit': loss_betz,
            'cut_in': loss_cut_in,
            'cut_out': loss_cut_out,
            'monotonicity': loss_monotonicity
        }
    
    def uncertainty_estimate(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout for uncertainty quantification."""
        self.train()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                power, _ = self.forward(x, apply_constraints=True)
                predictions.append(power)
        
        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        self.eval()
        return mean, std
    
    def explain_prediction(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Gradient-based feature importance."""
        x.requires_grad_(True)
        power, _ = self.forward(x, apply_constraints=False)
        
        power.sum().backward()
        
        importance = torch.abs(x.grad * x)
        feature_names = ['Wind_Speed', 'Direction', 'Temperature', 'Pressure', 'Turbulence']
        
        result = {name: importance[:, i].mean().item() 
                 for i, name in enumerate(feature_names)}
        
        x.requires_grad_(False)
        return result

# --- Unit Test ---
if __name__ == "__main__":
    batch_size = 10
    
    # Test scenarios
    # Scenario 1: Below cut-in (2 m/s)
    # Scenario 2: Optimal wind (10 m/s)
    # Scenario 3: Above cut-out (30 m/s)
    
    wind_speeds = torch.tensor([2.0, 10.0, 30.0]).repeat(batch_size, 1).reshape(-1, 1)
    direction = torch.full_like(wind_speeds, 180.0)
    temp = torch.full_like(wind_speeds, 15.0)
    pressure = torch.full_like(wind_speeds, 101325.0)
    turbulence = torch.full_like(wind_speeds, 0.15)
    
    x = torch.cat([wind_speeds, direction, temp, pressure, turbulence], dim=-1)
    
    model = WindPINN(input_dim=5, rated_power_kw=2000.0, rotor_diameter_m=80.0)
    
    # Forward pass
    power_pred, C_p_pred = model(x)
    
    print("=" * 50)
    print("WIND PINN TEST")
    print("=" * 50)
    
    for i, v in enumerate([2.0, 10.0, 30.0]):
        idx = i * batch_size
        print(f"\nWind Speed: {v} m/s")
        print(f"  Power: {power_pred[idx].item():.2f} kW")
        print(f"  C_p: {C_p_pred[idx].item():.4f} (Betz limit: {model.physics.C_p_max:.4f})")
    
    # Physics loss
    x_test = x[:batch_size]
    x_test.requires_grad_(True)
    power_test, C_p_test = model(x_test, apply_constraints=False)
    
    physics_losses = model.physics_loss(x_test, power_test, C_p_test)
    print("\nPhysics Losses:")
    for key, val in physics_losses.items():
        print(f"  {key}: {val.item():.6f}")
    
    # Uncertainty
    mean_power, std_power = model.uncertainty_estimate(x[:3], n_samples=10)
    print(f"\nUncertainty (first sample): {mean_power[0].item():.2f} ± {std_power[0].item():.2f} kW")
    
    # Explainability
    importance = model.explain_prediction(x[:3])
    print("\nFeature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feature}: {score:.4f}")
