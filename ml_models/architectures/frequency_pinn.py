"""
Grid Frequency Dynamics Model - Swing Equation Solver.
Implements physics-informed prediction of grid frequency response to disturbances.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class FrequencyPhysicsConstraints(nn.Module):
    """
    Enforces power system frequency dynamics:
    1. Swing Equation: 2H * df/dt = P_mech - P_elec - D * Δf
    2. Frequency bounds: 59.5 Hz <= f <= 60.5 Hz (or 49.5-50.5 for 50Hz systems)
    3. Rate of Change of Frequency (RoCoF) limits
    """
    def __init__(self, nominal_freq: float = 60.0, inertia_constant_H: float = 5.0):
        super().__init__()
        self.f_nom = nominal_freq  # Hz
        self.H = inertia_constant_H  # Inertia constant (seconds)
        self.D = 1.0  # Damping coefficient (per unit)
        
        # Operational limits
        self.f_min = nominal_freq - 0.5
        self.f_max = nominal_freq + 0.5
        
        # RoCoF limit (Hz/s)
        self.rocof_max = 1.0  # Typical limit for grid stability
        
    def swing_equation_residual(self, freq_t: torch.Tensor, freq_next: torch.Tensor,
                                p_gen: torch.Tensor, p_load: torch.Tensor, 
                                dt: float = 1.0) -> torch.Tensor:
        """
        Physics-informed swing equation:
        2H * (f_{t+1} - f_t) / dt = P_gen - P_load - D * (f_t - f_nom)
        
        Where:
        - H: Inertia constant (seconds)
        - P_gen: Generation power (per unit)
        - P_load: Load power (per unit)
        - D: Damping coefficient
        """
        freq_delta = freq_next - freq_t
        delta_f = freq_t - self.f_nom
        
        # Left side: Inertial response
        inertial_term = (2 * self.H * freq_delta) / dt
        
        # Right side: Power imbalance and damping
        power_imbalance = p_gen - p_load - self.D * delta_f
        
        residual = inertial_term - power_imbalance
        return torch.mean(residual ** 2)
    
    def frequency_bounds_loss(self, freq: torch.Tensor) -> torch.Tensor:
        """Soft penalty for frequency deviations outside operational limits."""
        violation_low = torch.clamp(self.f_min - freq, min=0.0)
        violation_high = torch.clamp(freq - self.f_max, min=0.0)
        return torch.mean(violation_low ** 2 + violation_high ** 2)
    
    def rocof_constraint_loss(self, freq_t: torch.Tensor, freq_next: torch.Tensor, 
                             dt: float = 1.0) -> torch.Tensor:
        """
        Penalize excessive Rate of Change of Frequency (RoCoF).
        Sudden frequency changes indicate loss of generation or load.
        """
        rocof = torch.abs(freq_next - freq_t) / dt
        violation = torch.clamp(rocof - self.rocof_max, min=0.0)
        return torch.mean(violation ** 2)
    
    def project_frequency(self, freq: torch.Tensor) -> torch.Tensor:
        """Hard constraint projection to valid frequency range."""
        return torch.clamp(freq, min=self.f_min, max=self.f_max)

class FrequencyPINN(nn.Module):
    """
    Physics-Informed Neural Network for Grid Frequency Prediction.
    
    Inputs: [Freq_t, P_gen, P_load, RoCoF_t, Inertia_available]
    Outputs: [Freq_{t+1}, RoCoF_{t+1}]
    """
    def __init__(self, input_dim: int = 5, nominal_freq: float = 60.0, 
                 inertia_H: float = 5.0):
        super().__init__()
        
        # Neural Network Backbone
        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2)  # [Freq_next, RoCoF_next]
        )
        
        # Physics Constraints
        self.physics = FrequencyPhysicsConstraints(nominal_freq, inertia_H)
        
        # Uncertainty
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
            x: [Batch, 5] -> [Freq_t, P_gen, P_load, RoCoF_t, H_available]
        Returns:
            freq_next: Predicted frequency at t+1
            rocof_next: Predicted RoCoF
        """
        output = self.dynamics_net(self.dropout(x))
        
        freq_raw = output[:, 0:1]
        rocof_raw = output[:, 1:2]
        
        if apply_constraints:
            # Project frequency to valid range
            freq_next = self.physics.project_frequency(freq_raw)
            rocof_next = rocof_raw  # RoCoF can be positive or negative
        else:
            freq_next = freq_raw
            rocof_next = rocof_raw
            
        return freq_next, rocof_next
    
    def physics_loss(self, x: torch.Tensor, freq_next: torch.Tensor, 
                    rocof_next: torch.Tensor, dt: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Computes physics-informed loss components.
        """
        freq_t = x[:, 0:1]
        p_gen = x[:, 1:2]
        p_load = x[:, 2:3]
        rocof_t = x[:, 3:4]
        
        # Loss 1: Swing Equation Residual
        loss_swing = self.physics.swing_equation_residual(
            freq_t, freq_next, p_gen, p_load, dt
        )
        
        # Loss 2: Frequency Bounds
        loss_bounds = self.physics.frequency_bounds_loss(freq_next)
        
        # Loss 3: RoCoF Constraint
        loss_rocof = self.physics.rocof_constraint_loss(freq_t, freq_next, dt)
        
        # Loss 4: RoCoF Consistency (RoCoF = df/dt)
        rocof_from_freq = (freq_next - freq_t) / dt
        loss_rocof_consistency = torch.mean((rocof_next - rocof_from_freq) ** 2)
        
        # Loss 5: Power Balance (for equilibrium, df/dt = 0 when P_gen = P_load)
        # At equilibrium, frequency should stabilize
        equilibrium_mask = (torch.abs(p_gen - p_load) < 0.01).float()
        loss_equilibrium = torch.mean(equilibrium_mask * (freq_next - self.physics.f_nom) ** 2)
        
        return {
            'swing_equation': loss_swing,
            'frequency_bounds': loss_bounds,
            'rocof_limit': loss_rocof,
            'rocof_consistency': loss_rocof_consistency,
            'equilibrium': loss_equilibrium
        }
    
    def compute_physics_derivatives(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute higher-order derivatives for enhanced physics supervision.
        Useful for verifying that d²f/dt² follows expected dynamics.
        """
        x.requires_grad_(True)
        freq_next, rocof_next = self.forward(x, apply_constraints=False)
        
        # First derivative: df/dt (RoCoF)
        df_dx = torch.autograd.grad(
            outputs=freq_next,
            inputs=x,
            grad_outputs=torch.ones_like(freq_next),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative: d²f/dt²
        d2f_dx2 = torch.autograd.grad(
            outputs=df_dx.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        x.requires_grad_(False)
        
        return {
            'df_dx': df_dx,
            'd2f_dx2': d2f_dx2
        }
    
    def uncertainty_estimate(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout for uncertainty quantification."""
        self.train()
        freq_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                freq, _ = self.forward(x, apply_constraints=True)
                freq_samples.append(freq)
        
        freq_samples = torch.stack(freq_samples, dim=0)
        mean = freq_samples.mean(dim=0)
        std = freq_samples.std(dim=0)
        
        self.eval()
        return mean, std
    
    def explain_prediction(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Gradient-based feature importance."""
        x.requires_grad_(True)
        freq_next, _ = self.forward(x, apply_constraints=False)
        
        freq_next.sum().backward()
        
        importance = torch.abs(x.grad * x)
        feature_names = ['Freq_t', 'P_gen', 'P_load', 'RoCoF_t', 'H_available']
        
        result = {name: importance[:, i].mean().item() 
                 for i, name in enumerate(feature_names)}
        
        x.requires_grad_(False)
        return result
    
    def stability_assessment(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Assess grid stability based on predicted frequency response.
        Returns stability metrics.
        """
        freq_next, rocof_next = self.forward(x, apply_constraints=True)
        freq_t = x[:, 0:1]
        
        # Metric 1: Frequency deviation from nominal
        freq_deviation = torch.abs(freq_next - self.physics.f_nom)
        
        # Metric 2: Rate of change magnitude
        rocof_magnitude = torch.abs(rocof_next)
        
        # Metric 3: Stability indicator (low deviation + low RoCoF = stable)
        stability_score = 1.0 / (1.0 + freq_deviation + 0.5 * rocof_magnitude)
        
        return {
            'frequency_deviation_hz': freq_deviation,
            'rocof_hz_per_s': rocof_magnitude,
            'stability_score': stability_score
        }

# --- Unit Test ---
if __name__ == "__main__":
    batch_size = 5
    
    # Test scenarios
    # Scenario 1: Balanced system (P_gen = P_load)
    # Scenario 2: Generation loss (P_gen < P_load)
    # Scenario 3: Load loss (P_gen > P_load)
    
    freq_t = torch.tensor([60.0, 60.0, 60.0, 59.8, 60.2]).unsqueeze(1)
    p_gen = torch.tensor([1.0, 0.8, 1.2, 0.9, 1.1]).unsqueeze(1)  # per unit
    p_load = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).unsqueeze(1)
    rocof_t = torch.tensor([0.0, -0.2, 0.2, -0.3, 0.1]).unsqueeze(1)
    H_avail = torch.tensor([5.0, 5.0, 5.0, 4.0, 5.5]).unsqueeze(1)
    
    x = torch.cat([freq_t, p_gen, p_load, rocof_t, H_avail], dim=-1)
    
    model = FrequencyPINN(input_dim=5, nominal_freq=60.0, inertia_H=5.0)
    
    # Forward pass
    freq_next, rocof_next = model(x)
    
    print("=" * 50)
    print("FREQUENCY PINN TEST")
    print("=" * 50)
    
    for i in range(batch_size):
        print(f"\nScenario {i+1}:")
        print(f"  Freq: {freq_t[i].item():.3f} Hz -> {freq_next[i].item():.3f} Hz")
        print(f"  Power: Gen={p_gen[i].item():.2f} pu, Load={p_load[i].item():.2f} pu")
        print(f"  RoCoF: {rocof_t[i].item():.3f} -> {rocof_next[i].item():.3f} Hz/s")
    
    # Physics loss
    physics_losses = model.physics_loss(x, freq_next, rocof_next, dt=1.0)
    print("\nPhysics Losses:")
    for key, val in physics_losses.items():
        print(f"  {key}: {val.item():.6f}")
    
    # Stability assessment
    stability = model.stability_assessment(x)
    print("\nStability Assessment:")
    for i in range(batch_size):
        print(f"  Scenario {i+1}: Score={stability['stability_score'][i].item():.3f}, "
              f"Dev={stability['frequency_deviation_hz'][i].item():.3f} Hz, "
              f"RoCoF={stability['rocof_hz_per_s'][i].item():.3f} Hz/s")
    
    # Uncertainty
    mean_freq, std_freq = model.uncertainty_estimate(x[:2], n_samples=10)
    print(f"\nUncertainty (first sample): {mean_freq[0].item():.3f} ± {std_freq[0].item():.3f} Hz")
    
    # Explainability
    importance = model.explain_prediction(x[:2])
    print("\nFeature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feature}: {score:.4f}")
