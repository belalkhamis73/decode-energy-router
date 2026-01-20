"""
Wind Turbine Power Model Trainer
Physics-Informed Neural Network for Wind Energy Prediction

TRAINING DATA REQUIREMENTS:
- Wind speed at hub height [m/s]
- Actual turbine output measurements [kW]
- Air density [kg/m³] (optional, can be calculated from temp/pressure)
- Cut-in/cut-out/rated wind speeds labeled
- Turbine specifications (rotor diameter, rated power)
- Wind direction (optional for wake effects)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

from ml_models.architectures.deeponet import DeepONet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WindModelTrainer")


class WindPhysicsLoss(nn.Module):
    """
    Physics-Informed Loss enforcing wind turbine power curve equations.
    
    Key Physics Constraints:
    1. P = 0 when v < v_cut_in or v > v_cut_out
    2. P = 0.5 * ρ * A * Cp * v³ in operational range
    3. P = P_rated when v_rated ≤ v < v_cut_out
    4. Power curve monotonicity between cut-in and rated
    5. Betz limit: Cp ≤ 0.59
    """
    
    def __init__(self, rated_power: float, rotor_diameter: float,
                 v_cut_in: float = 3.0, v_rated: float = 12.0, 
                 v_cut_out: float = 25.0, air_density: float = 1.225):
        super().__init__()
        self.P_rated = rated_power  # kW
        self.A = np.pi * (rotor_diameter / 2) ** 2  # Swept area m²
        self.rho = air_density  # kg/m³
        self.v_cut_in = v_cut_in
        self.v_rated = v_rated
        self.v_cut_out = v_cut_out
        self.Cp_max = 0.45  # Typical max power coefficient
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                wind_speed: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: [batch, 1] predicted power
            targets: [batch, 1] actual measurements
            wind_speed: [batch] wind speeds in m/s
        """
        # Data Loss
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # Physics Constraints
        v = wind_speed.unsqueeze(-1)  # [batch, 1]
        
        # 1. Cut-in/Cut-out constraints
        below_cutin = (v < self.v_cut_in).float()
        above_cutout = (v > self.v_cut_out).float()
        shutdown_mask = below_cutin + above_cutout
        shutdown_loss = (predictions * shutdown_mask).abs().mean()
        
        # 2. Theoretical power from wind (cubic relationship)
        theoretical_power = (0.5 * self.rho * self.A * self.Cp_max * 
                           torch.pow(v, 3) / 1000.0)  # Convert to kW
        theoretical_power = theoretical_power.clamp(0, self.P_rated)
        
        # Only enforce in operational range
        operational_mask = ((v >= self.v_cut_in) & (v < self.v_cut_out)).float()
        physics_loss = (operational_mask * (predictions - theoretical_power) ** 2).mean()
        
        # 3. Rated power constraint
        above_rated = (v >= self.v_rated).float() * (v < self.v_cut_out).float()
        rated_loss = (above_rated * (predictions - self.P_rated) ** 2).mean()
        
        # 4. Capacity constraint
        capacity_violation = torch.relu(predictions - self.P_rated)
        capacity_loss = capacity_violation.mean()
        
        # 5. Non-negativity
        negativity_loss = torch.relu(-predictions).mean()
        
        # Total Loss
        total_loss = (
            mse_loss +
            0.3 * physics_loss +
            1.0 * shutdown_loss +
            0.5 * rated_loss +
            2.0 * capacity_loss +
            5.0 * negativity_loss
        )
        
        metrics = {
            'mse': mse_loss.item(),
            'physics': physics_loss.item(),
            'shutdown_violation': shutdown_loss.item(),
            'rated_deviation': rated_loss.item(),
            'capacity_violation': capacity_loss.item(),
            'negativity': negativity_loss.item()
        }
        
        return total_loss, metrics


def train_wind_model(
    dataset: Dict[str, Any],
    hyperparams: Dict[str, Any],
    pretrained_path: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints/wind"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train Wind Turbine power prediction model.
    
    Args:
        dataset: {
            'wind_speed': [N, 24], 'actual_output': [N, 24],
            'air_density': [N, 24] (optional),
            'wind_direction': [N, 24] (optional),
            'metadata': {
                'rated_power': float, 'rotor_diameter': float,
                'v_cut_in': float, 'v_rated': float, 'v_cut_out': float
            }
        }
        hyperparams: Training configuration
        pretrained_path: Path to pretrained checkpoint
        checkpoint_dir: Save directory
        
    Returns:
        model: Trained model
        summary: Training metrics
    """
    
    # Setup
    device = hyperparams.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    epochs = hyperparams.get('epochs', 100)
    lr = hyperparams.get('learning_rate', 1e-3)
    batch_size = hyperparams.get('batch_size', 32)
    patience = hyperparams.get('early_stop_patience', 15)
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🌬️ Initializing Wind Model Training on {device}")
    
    # Data Preparation
    wind_speed = torch.tensor(dataset['wind_speed'], dtype=torch.float32)
    targets = torch.tensor(dataset['actual_output'], dtype=torch.float32)
    
    # Optional features
    if 'air_density' in dataset:
        air_density = torch.tensor(dataset['air_density'], dtype=torch.float32)
    else:
        air_density = torch.ones_like(wind_speed) * 1.225
    
    if 'wind_direction' in dataset:
        wind_dir = torch.tensor(dataset['wind_direction'], dtype=torch.float32)
        features = torch.stack([wind_speed, air_density, wind_dir], dim=-1)
        input_dim = 3
    else:
        features = torch.stack([wind_speed, air_density], dim=-1)
        input_dim = 2
    
    # Reshape: [N, 24, features] -> [N*24, features]
    N = features.shape[0]
    features = features.reshape(-1, input_dim)
    targets = targets.reshape(-1, 1)
    wind_speed_flat = wind_speed.reshape(-1)
    
    # Train/Val Split
    split_idx = int(0.8 * len(features))
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = targets[:split_idx], targets[split_idx:]
    train_ws, val_ws = wind_speed_flat[:split_idx], wind_speed_flat[split_idx:]
    
    # Model
    model = DeepONet(
        input_dim=input_dim,
        hidden_dim=hyperparams.get('hidden_dim', 128),
        output_dim=1,
        n_buses=1
    ).to(device)
    
    # Transfer Learning
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"🔄 Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Physics Loss
    metadata = dataset.get('metadata', {})
    physics_loss = WindPhysicsLoss(
        rated_power=metadata.get('rated_power', 2000.0),  # kW
        rotor_diameter=metadata.get('rotor_diameter', 80.0),  # m
        v_cut_in=metadata.get('v_cut_in', 3.0),
        v_rated=metadata.get('v_rated', 12.0),
        v_cut_out=metadata.get('v_cut_out', 25.0)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'physics_metrics': []}
    
    logger.info(f"🚀 Training for {epochs} epochs")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size].to(device)
            batch_y = train_y[i:i+batch_size].to(device)
            batch_ws = train_ws[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            dummy_buses = torch.zeros(len(batch_x), 1, dtype=torch.long, device=device)
            predictions = model(batch_x, dummy_buses)
            
            loss, metrics = physics_loss(predictions, batch_y, batch_ws)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x_gpu = val_x.to(device)
            val_y_gpu = val_y.to(device)
            val_ws_gpu = val_ws.to(device)
            dummy_buses_val = torch.zeros(len(val_x), 1, dtype=torch.long, device=device)
            
            val_preds = model(val_x_gpu, dummy_buses_val)
            val_loss, val_metrics = physics_loss(val_preds, val_y_gpu, val_ws_gpu)
        
        avg_train_loss = np.mean(train_losses)
        scheduler.step()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['physics_metrics'].append(val_metrics)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss.item():.4f}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss.item(),
                'hyperparams': hyperparams,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_path = Path(checkpoint_dir) / f"wind_best_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"⏸️ Early stopping at epoch {epoch}")
            break
    
    logger.info(f"✅ Training Complete | Best Val Loss: {best_val_loss:.4f}")
    
    summary = {
        'final_train_loss': history['train_loss'][-1],
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'history': history,
        'model_type': 'wind_turbine',
        'physics_validation': history['physics_metrics'][-1]
    }
    
    return model.cpu(), summary


if __name__ == "__main__":
    # Unit Test
    mock_data = {
        'wind_speed': np.random.rand(100, 24) * 20,
        'actual_output': np.random.rand(100, 24) * 1800,
        'metadata': {
            'rated_power': 2000,
            'rotor_diameter': 80,
            'v_cut_in': 3.0,
            'v_rated': 12.0,
            'v_cut_out': 25.0
        }
    }
    
    params = {'epochs': 20, 'learning_rate': 1e-3, 'batch_size': 16}
    
    try:
        model, summary = train_wind_model(mock_data, params)
        print(f"✅ Test Successful: {summary['best_val_loss']:.4f}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise
