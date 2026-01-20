"""
Load Forecasting Model with LSTM + Attention Mechanism.
Predicts electrical load demand with temporal awareness and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class AttentionLayer(nn.Module):
    """
    Scaled Dot-Product Attention for time series.
    Allows the model to focus on relevant historical timesteps.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, Seq_Len, Hidden_Dim]
        Returns:
            attended: Weighted combination of values
            weights: Attention weights for interpretability
        """
        Q = self.query(x)  # [B, T, H]
        K = self.key(x)    # [B, T, H]
        V = self.value(x)  # [B, T, H]
        
        # Attention scores: Q * K^T / sqrt(d_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, T, T]
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        attended = torch.bmm(weights, V)  # [B, T, H]
        
        return attended, weights

class LoadForecastingConstraints(nn.Module):
    """
    Enforces load forecasting constraints:
    1. Non-negativity: Load >= 0
    2. Physical bounds: Load <= Peak_Capacity
    3. Temporal smoothness: Avoid unrealistic jumps
    4. Periodic patterns: Daily/weekly cycles
    """
    def __init__(self, peak_load_mw: float = 1000.0):
        super().__init__()
        self.peak_load = peak_load_mw
        
    def non_negativity_loss(self, load: torch.Tensor) -> torch.Tensor:
        """Penalize negative load predictions."""
        violation = torch.clamp(-load, min=0.0)
        return torch.mean(violation ** 2)
    
    def capacity_bounds_loss(self, load: torch.Tensor) -> torch.Tensor:
        """Penalize predictions exceeding peak capacity."""
        violation = torch.clamp(load - self.peak_load, min=0.0)
        return torch.mean(violation ** 2)
    
    def temporal_smoothness_loss(self, load_sequence: torch.Tensor) -> torch.Tensor:
        """
        Penalize large jumps between consecutive timesteps.
        Load typically changes gradually except for special events.
        """
        # Compute first-order differences
        diff = load_sequence[:, 1:] - load_sequence[:, :-1]
        
        # Penalize large changes (using Huber-like loss)
        threshold = 50.0  # MW - typical acceptable hourly change
        smooth_loss = torch.where(
            torch.abs(diff) < threshold,
            0.5 * diff ** 2,
            threshold * (torch.abs(diff) - 0.5 * threshold)
        )
        
        return torch.mean(smooth_loss)
    
    def project_load(self, load: torch.Tensor) -> torch.Tensor:
        """Hard constraint: 0 <= Load <= Peak"""
        return torch.clamp(load, min=0.0, max=self.peak_load)

class LoadForecastingModel(nn.Module):
    """
    LSTM + Attention Model for Load Forecasting.
    
    Inputs: [Temperature, Hour, Day_of_Week, Month, Is_Holiday, Historical_Load, ...]
    Outputs: [Load_t+1, Load_t+2, ..., Load_t+H] (multi-step forecast)
    """
    def __init__(self, input_dim: int = 7, hidden_dim: int = 128, 
                 num_layers: int = 2, forecast_horizon: int = 24,
                 peak_load_mw: float = 1000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Attention Mechanism
        self.attention = AttentionLayer(hidden_dim)
        
        # Forecasting Head
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)
        )
        
        # Constraints
        self.constraints = LoadForecastingConstraints(peak_load_mw)
        
        # Uncertainty estimation (aleatoric + epistemic)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon)  # Predicts log_variance
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, apply_constraints: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, Seq_Len, Features]
        Returns:
            load_forecast: [Batch, Horizon] - Predicted load
            uncertainty: [Batch, Horizon] - Prediction uncertainty (std)
            attention_weights: [Batch, Seq_Len, Seq_Len] - Attention map
        """
        # Encode temporal sequence
        lstm_out, _ = self.lstm(x)  # [B, T, H]
        
        # Apply attention
        attended, attn_weights = self.attention(lstm_out)
        
        # Use last timestep's attended representation
        context = attended[:, -1, :]  # [B, H]
        
        # Forecast future load
        load_raw = self.forecast_head(context)  # [B, Horizon]
        
        # Estimate uncertainty (aleatoric)
        log_var = self.uncertainty_head(context)
        uncertainty = torch.exp(0.5 * log_var)  # std = exp(0.5 * log_var)
        
        if apply_constraints:
            load_forecast = self.constraints.project_load(load_raw)
        else:
            load_forecast = load_raw
            
        return load_forecast, uncertainty, attn_weights
    
    def physics_loss(self, x: torch.Tensor, load_forecast: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes physics-informed and constraint losses.
        """
        # Loss 1: Non-negativity
        loss_non_neg = self.constraints.non_negativity_loss(load_forecast)
        
        # Loss 2: Capacity bounds
        loss_capacity = self.constraints.capacity_bounds_loss(load_forecast)
        
        # Loss 3: Temporal smoothness
        loss_smoothness = self.constraints.temporal_smoothness_loss(load_forecast)
        
        # Loss 4: Periodicity consistency (daily pattern)
        # If we have 24-hour forecast, first and last hour should be similar (daily cycle)
        if load_forecast.shape[1] >= 24:
            loss_periodicity = torch.mean((load_forecast[:, 0] - load_forecast[:, 23]) ** 2)
        else:
            loss_periodicity = torch.tensor(0.0)
        
        return {
            'non_negativity': loss_non_neg,
            'capacity_bounds': loss_capacity,
            'temporal_smoothness': loss_smoothness,
            'periodicity': loss_periodicity
        }
    
    def compute_loss_with_uncertainty(self, load_pred: torch.Tensor, load_target: torch.Tensor,
                                     uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss that accounts for predicted uncertainty.
        Model learns to be more uncertain when predictions are difficult.
        
        NLL = 0.5 * log(2π * σ²) + (y - ŷ)² / (2σ²)
        """
        mse = (load_pred - load_target) ** 2
        var = uncertainty ** 2 + 1e-6  # Add epsilon for stability
        
        nll = 0.5 * torch.log(var) + mse / (2 * var)
        return torch.mean(nll)
    
    def uncertainty_estimate_mcdropout(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for epistemic uncertainty.
        Combines with aleatoric uncertainty for total uncertainty.
        """
        self.train()  # Enable dropout
        
        predictions = []
        aleatoric_vars = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                load, aleatoric_std, _ = self.forward(x, apply_constraints=True)
                predictions.append(load)
                aleatoric_vars.append(aleatoric_std ** 2)
        
        predictions = torch.stack(predictions, dim=0)  # [N, B, H]
        aleatoric_vars = torch.stack(aleatoric_vars, dim=0)
        
        # Epistemic uncertainty: variance across predictions
        epistemic_var = predictions.var(dim=0)
        
        # Aleatoric uncertainty: average of predicted variances
        aleatoric_var = aleatoric_vars.mean(dim=0)
        
        # Total uncertainty
        total_std = torch.sqrt(epistemic_var + aleatoric_var)
        
        mean = predictions.mean(dim=0)
        
        self.eval()
        return mean, total_std
    
    def explain_prediction(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Multi-level explainability:
        1. Attention weights (which timesteps matter most)
        2. Gradient-based feature importance
        """
        x.requires_grad_(True)
        load_forecast, _, attn_weights = self.forward(x, apply_constraints=False)
        
        # Compute gradients w.r.t. input features
        load_forecast.sum().backward()
        
        # Feature importance: |gradient| * |input|
        importance = torch.abs(x.grad * x)
        
        # Temporal importance: Sum importance across features for each timestep
        temporal_importance = importance.sum(dim=-1)  # [B, T]
        
        # Feature importance: Average across timesteps
        feature_importance = importance.mean(dim=1)  # [B, F]
        
        x.requires_grad_(False)
        
        return {
            'attention_weights': attn_weights,  # [B, T, T]
            'temporal_importance': temporal_importance,  # [B, T]
            'feature_importance': feature_importance  # [B, F]
        }
    
    def extract_attention_patterns(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze attention patterns to understand model's temporal focus.
        Useful for debugging and model interpretation.
        """
        _, _, attn_weights = self.forward(x, apply_constraints=True)
        
        # Average attention across batch
        avg_attention = attn_weights.mean(dim=0)  # [T, T]
        
        # Attention to most recent vs. distant past
        recent_focus = avg_attention[:, -3:].mean()  # Focus on last 3 timesteps
        distant_focus = avg_attention[:, :-3].mean()  # Focus on earlier timesteps
        
        return {
            'attention_matrix': avg_attention,
            'recent_focus': recent_focus,
            'distant_focus': distant_focus
        }

# --- Unit Test ---
if __name__ == "__main__":
    batch_size = 4
    seq_len = 48  # 48 hours of history
    input_dim = 7  # [Temp, Hour, Day, Month, Holiday, Lag_Load, Lag_Load_24h]
    forecast_horizon = 24  # Predict next 24 hours
    
    # Synthetic data: Temperature cycles + time features + historical load
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Add realistic patterns
    for i in range(seq_len):
        hour = i % 24
        x[:, i, 1] = hour / 24.0  # Normalized hour
        x[:, i, 2] = (i // 24) % 7 / 7.0  # Day of week
        x[:, i, 0] = 20 + 10 * np.sin(2 * np.pi * hour / 24)  # Temperature cycle
    
    model = LoadForecastingModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        forecast_horizon=forecast_horizon,
        peak_load_mw=1000.0
    )
    
    # Forward pass
    load_forecast, uncertainty, attn_weights = model(x)
    
    print("=" * 50)
    print("LOAD FORECASTING MODEL TEST")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    print(f"Forecast shape: {load_forecast.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"\nSample forecast (next 24 hours):")
    print(f"  Load: {load_forecast[0, :6].detach().numpy()}")
    print(f"  Uncertainty: {uncertainty[0, :6].detach().numpy()}")
    
    # Physics losses
    physics_losses = model.physics_loss(x, load_forecast)
    print("\nPhysics Losses:")
    for key, val in physics_losses.items():
        print(f"  {key}: {val.item():.6f}")
    
    # MC Dropout uncertainty
    mean_load, total_std = model.uncertainty_estimate_mcdropout(x[:2], n_samples=10)
    print(f"\nTotal Uncertainty (first sample, t+1):")
    print(f"  {mean_load[0, 0].item():.2f} ± {total_std[0, 0].item():.2f} MW")
    
    # Explainability
    explainability = model.explain_prediction(x[:2])
    print(f"\nFeature Importance (averaged):")
    feature_names = ['Temperature', 'Hour', 'Day', 'Month', 'Holiday', 'Lag_Load', 'Lag_Load_24h']
    for i, name in enumerate(feature_names):
        print(f"  {name}: {explainability['feature_importance'][0, i].item():.4f}")
    
    # Attention patterns
    attn_patterns = model.extract_attention_patterns(x)
    print(f"\nAttention Focus:")
    print(f"  Recent (last 3 hours): {attn_patterns['recent_focus'].item():.4f}")
    print(f"  Distant (earlier): {attn_patterns['distant_focus'].item():.4f}")
