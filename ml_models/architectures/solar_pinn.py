"""
Solar PV Production Model with Irradiance Physics.
Enforces solar radiation transfer equations and panel efficiency curves.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class SolarPhysicsConstraints(nn.Module):
    """
    Enforces fundamental solar physics:
    1. GHI = DNI * cos(zenith) + DHI (irradiance decomposition)
    2. Panel efficiency curve (temperature-dependent)
    3. Bounding: 0 <= Power <= Rated_Capacity
    """
    def __init__(self, panel_efficiency: float = 0.20, temp_coeff: float = -0.004):
        super().__init__()
        self.eta_stc = panel_efficiency  # Efficiency at Standard Test Conditions
        self.beta = temp_coeff  # Temperature coefficient (%/°C)
        self.T_stc = 25.0  # Standard Test Condition temperature (°C)
        
    def irradiance_decomposition_loss(self, dni: torch.Tensor, dhi: torch.Tensor, 
                                     ghi_pred: torch.Tensor, cos_zenith: torch.Tensor) -> torch.Tensor:
        """
        Physics Loss: GHI = DNI * cos(zenith) + DHI
        This is a fundamental atmospheric radiative transfer equation.
        """
        ghi_physical = dni * torch.clamp(cos_zenith, min=0.0) + dhi
        return torch.mean((ghi_pred - ghi_physical) ** 2)
    
    def efficiency_curve(self, ghi: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        """
        Temperature-dependent efficiency:
        eta = eta_STC * [1 + beta * (T_cell - T_STC)]
        Where T_cell ≈ T_ambient + k * GHI (empirical cell heating)
        """
        k_noct = 0.02  # NOCT factor (empirical: ~0.02 °C per W/m²)
        T_cell = temp + k_noct * ghi
        eta = self.eta_stc * (1.0 + self.beta * (T_cell - self.T_stc))
        return torch.clamp(eta, min=0.05, max=0.25)  # Physical bounds
    
    def project_power_bounds(self, power: torch.Tensor, capacity: float) -> torch.Tensor:
        """Enforce 0 <= P <= P_rated"""
        return torch.clamp(power, min=0.0, max=capacity)

class SolarPINN(nn.Module):
    """
    Physics-Informed Neural Network for Solar PV Production.
    
    Inputs: [DNI, DHI, Temperature, Cos(Zenith_Angle), Hour_of_Day] 
    Outputs: [GHI_predicted, Power_AC]
    """
    def __init__(self, input_dim: int = 5, capacity_kw: float = 100.0):
        super().__init__()
        self.capacity = capacity_kw
        
        # Neural Network Backbone
        self.irradiance_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Predicts GHI
        )
        
        self.power_net = nn.Sequential(
            nn.Linear(input_dim + 1, 32),  # +1 for predicted GHI
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # Predicts Power
        )
        
        # Physics Constraints
        self.physics = SolarPhysicsConstraints()
        
        # Uncertainty estimation (Bayesian approximation via dropout)
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
            x: [Batch, 5] -> [DNI, DHI, Temp, Cos_Zenith, Hour]
        Returns:
            ghi_pred: Predicted Global Horizontal Irradiance
            power_pred: Predicted AC Power Output
        """
        # Step 1: Predict GHI from atmospheric inputs
        ghi_pred = self.irradiance_net(self.dropout(x))
        
        # Step 2: Concatenate GHI with inputs for power prediction
        power_input = torch.cat([x, ghi_pred], dim=-1)
        power_raw = self.power_net(self.dropout(power_input))
        
        # Step 3: Apply physics-based constraints
        if apply_constraints:
            power_pred = self.physics.project_power_bounds(power_raw, self.capacity)
        else:
            power_pred = power_raw
            
        return ghi_pred, power_pred
    
    def physics_loss(self, x: torch.Tensor, ghi_pred: torch.Tensor, 
                    power_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes physics-informed loss components.
        """
        dni = x[:, 0:1]
        dhi = x[:, 1:2]
        temp = x[:, 2:3]
        cos_zenith = x[:, 3:4]
        
        # Loss 1: Irradiance Decomposition (hard physics)
        loss_irradiance = self.physics.irradiance_decomposition_loss(
            dni, dhi, ghi_pred, cos_zenith
        )
        
        # Loss 2: Efficiency Curve Consistency
        eta_expected = self.physics.efficiency_curve(ghi_pred, temp)
        power_from_physics = ghi_pred * eta_expected * self.capacity / 1000.0  # Convert to kW
        loss_efficiency = torch.mean((power_pred - power_from_physics) ** 2)
        
        # Loss 3: Night-time constraint (cos_zenith <= 0 => Power = 0)
        night_mask = (cos_zenith <= 0).float()
        loss_night = torch.mean(night_mask * power_pred ** 2)
        
        return {
            'irradiance_physics': loss_irradiance,
            'efficiency_physics': loss_efficiency,
            'night_constraint': loss_night
        }
    
    def uncertainty_estimate(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for uncertainty quantification.
        Returns mean and std of predictions.
        """
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                _, power = self.forward(x, apply_constraints=True)
                predictions.append(power)
        
        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        self.eval()
        return mean, std
    
    def explain_prediction(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Gradient-based feature importance (Integrated Gradients approximation).
        Shows which input features most influence the prediction.
        """
        x.requires_grad_(True)
        _, power = self.forward(x, apply_constraints=False)
        
        # Compute gradients w.r.t. inputs
        power.sum().backward()
        
        # Feature importance = |gradient| * |input|
        importance = torch.abs(x.grad * x)
        feature_names = ['DNI', 'DHI', 'Temperature', 'Cos_Zenith', 'Hour']
        
        result = {name: importance[:, i].mean().item() 
                 for i, name in enumerate(feature_names)}
        
        x.requires_grad_(False)
        return result

# --- Unit Test ---
if __name__ == "__main__":
    # Test with synthetic data
    batch_size = 10
    
    # Daytime scenario: DNI=800, DHI=100, Temp=30°C, Zenith=30°, Hour=12
    dni = torch.full((batch_size, 1), 800.0)
    dhi = torch.full((batch_size, 1), 100.0)
    temp = torch.full((batch_size, 1), 30.0)
    cos_zenith = torch.full((batch_size, 1), np.cos(np.radians(30)))
    hour = torch.full((batch_size, 1), 12.0)
    
    x = torch.cat([dni, dhi, temp, cos_zenith, hour], dim=-1)
    
    model = SolarPINN(input_dim=5, capacity_kw=100.0)
    
    # Forward pass
    ghi_pred, power_pred = model(x)
    
    print("=" * 50)
    print("SOLAR PINN TEST")
    print("=" * 50)
    print(f"Predicted GHI: {ghi_pred.mean().item():.2f} W/m²")
    print(f"Predicted Power: {power_pred.mean().item():.2f} kW")
    
    # Physics loss
    physics_losses = model.physics_loss(x, ghi_pred, power_pred)
    print("\nPhysics Losses:")
    for key, val in physics_losses.items():
        print(f"  {key}: {val.item():.6f}")
    
    # Uncertainty
    mean_power, std_power = model.uncertainty_estimate(x, n_samples=10)
    print(f"\nUncertainty: {mean_power.mean().item():.2f} ± {std_power.mean().item():.2f} kW")
    
    # Explainability
    importance = model.explain_prediction(x)
    print("\nFeature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feature}: {score:.4f}")
