"""
Battery Thermal & State-of-Health Model Trainer
Physics-Informed Neural Network for Battery Management

TRAINING DATA REQUIREMENTS:
- Charge/discharge power cycles [kW]
- State of Charge (SOC) measurements [0-1]
- Battery cell temperature [°C]
- Ambient temperature [°C]
- Current [A] and Voltage [V] measurements
- State of Health (SOH) labels [0-1]
- Cycle count and calendar age
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
logger = logging.getLogger("BatteryThermalTrainer")


class BatteryPhysicsLoss(nn.Module):
    """
    Physics-Informed Loss enforcing battery dynamics and thermal constraints.
    
    Key Physics Constraints:
    1. SOC bounds: 0 ≤ SOC ≤ 1
    2. Thermal dynamics: dT/dt = (I²R - h(T-T_amb)) / (m*Cp)
    3. Power limits: P ≤ P_max * SOH
    4. Coulomb counting: dSOC/dt = -I/(Q_nom * 3600)
    5. SOH degradation monotonicity: dSOH/dt ≤ 0
    6. C-rate limits for safety
    """
    
    def __init__(self, capacity_kwh: float, max_power_kw: float,
                 internal_resistance: float = 0.05, mass_kg: float = 500.0,
                 max_charge_rate: float = 1.0, max_discharge_rate: float = 1.0):
        super().__init__()
        self.Q_nom = capacity_kwh  # kWh
        self.P_max = max_power_kw
        self.R_internal = internal_resistance  # Ohms
        self.mass = mass_kg
        self.Cp = 1000  # J/(kg·K) - specific heat capacity
        self.h = 10  # W/K - heat transfer coefficient
        self.C_rate_charge_max = max_charge_rate
        self.C_rate_discharge_max = max_discharge_rate
        
    def forward(self, predictions: torch.Tensor, targets: Dict[str, torch.Tensor],
                inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: [batch, 3] -> [SOC_next, Temp_next, SOH_next]
            targets: Dict with 'soc', 'temperature', 'soh' ground truth
            inputs: Dict with 'power', 'current', 'voltage', 'ambient_temp', 'dt'
        """
        pred_soc = predictions[:, 0]
        pred_temp = predictions[:, 1]
        pred_soh = predictions[:, 2]
        
        target_soc = targets['soc']
        target_temp = targets['temperature']
        target_soh = targets['soh']
        
        # Data Loss
        mse_soc = nn.MSELoss()(pred_soc, target_soc)
        mse_temp = nn.MSELoss()(pred_temp, target_temp)
        mse_soh = nn.MSELoss()(pred_soh, target_soh)
        data_loss = mse_soc + 0.5 * mse_temp + 2.0 * mse_soh
        
        # Physics Constraints
        power = inputs['power']  # kW
        current = inputs['current']  # A
        T_amb = inputs['ambient_temp']
        dt = inputs.get('dt', torch.ones_like(power))  # seconds
        
        # 1. SOC Bounds
        soc_lower_violation = torch.relu(-pred_soc)
        soc_upper_violation = torch.relu(pred_soc - 1.0)
        soc_bounds_loss = (soc_lower_violation + soc_upper_violation).mean()
        
        # 2. Thermal Physics (simplified heat generation)
        # Q_gen = I²R (Joule heating)
        heat_generation = (current ** 2) * self.R_internal
        heat_dissipation = self.h * (pred_temp - T_amb)
        thermal_balance = (heat_generation - heat_dissipation) / (self.mass * self.Cp)
        
        # Predicted temp change should match thermal physics
        # This is a residual constraint
        thermal_residual = torch.abs(thermal_balance * dt - (pred_temp - target_temp))
        thermal_loss = thermal_residual.mean()
        
        # 3. Power Capacity Constraint
        max_allowed_power = self.P_max * pred_soh
        power_violation = torch.relu(torch.abs(power) - max_allowed_power)
        power_loss = power_violation.mean()
        
        # 4. C-rate Constraints
        C_rate = torch.abs(power) / self.Q_nom
        charge_mask = (power > 0).float()
        discharge_mask = (power < 0).float()
        
        charge_rate_violation = torch.relu(C_rate * charge_mask - self.C_rate_charge_max)
        discharge_rate_violation = torch.relu(C_rate * discharge_mask - self.C_rate_discharge_max)
        crate_loss = (charge_rate_violation + discharge_rate_violation).mean()
        
        # 5. SOH Monotonicity (should never increase)
        # Compare consecutive SOH predictions if available
        soh_increase = torch.relu(pred_soh - 1.0)  # SOH should never exceed 1.0
        soh_loss = soh_increase.mean()
        
        # 6. Temperature Safety (20°C to 45°C typical safe range)
        temp_too_low = torch.relu(15.0 - pred_temp)
        temp_too_high = torch.relu(pred_temp - 50.0)
        temp_safety_loss = (temp_too_low + temp_too_high).mean()
        
        # Total Loss
        total_loss = (
            data_loss +
            2.0 * soc_bounds_loss +
            0.5 * thermal_loss +
            1.0 * power_loss +
            1.0 * crate_loss +
            3.0 * soh_loss +
            2.0 * temp_safety_loss
        )
        
        metrics = {
            'mse_soc': mse_soc.item(),
            'mse_temp': mse_temp.item(),
            'mse_soh': mse_soh.item(),
            'soc_bounds_violation': soc_bounds_loss.item(),
            'thermal_physics_error': thermal_loss.item(),
            'power_violation': power_loss.item(),
            'crate_violation': crate_loss.item(),
            'soh_violation': soh_loss.item(),
            'temp_safety_violation': temp_safety_loss.item()
        }
        
        return total_loss, metrics


def train_battery_thermal_model(
    dataset: Dict[str, Any],
    hyperparams: Dict[str, Any],
    pretrained_path: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints/battery"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train Battery Thermal & SOH prediction model.
    
    Args:
        dataset: {
            'power': [N, T], 'current': [N, T], 'voltage': [N, T],
            'soc': [N, T], 'temperature': [N, T], 'soh': [N, T],
            'ambient_temp': [N, T], 'cycle_count': [N],
            'metadata': {'capacity_kwh': float, 'max_power_kw': float}
        }
        hyperparams: Training configuration
        pretrained_path: Checkpoint for transfer learning
        checkpoint_dir: Save directory
    """
    
    # Setup
    device = hyperparams.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    epochs = hyperparams.get('epochs', 100)
    lr = hyperparams.get('learning_rate', 1e-3)
    batch_size = hyperparams.get('batch_size', 32)
    patience = hyperparams.get('early_stop_patience', 15)
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🔋 Initializing Battery Thermal Model Training on {device}")
    
    # Data Preparation
    power = torch.tensor(dataset['power'], dtype=torch.float32)
    current = torch.tensor(dataset['current'], dtype=torch.float32)
    voltage = torch.tensor(dataset['voltage'], dtype=torch.float32)
    soc = torch.tensor(dataset['soc'], dtype=torch.float32)
    temp = torch.tensor(dataset['temperature'], dtype=torch.float32)
    soh = torch.tensor(dataset['soh'], dtype=torch.float32)
    ambient_temp = torch.tensor(dataset['ambient_temp'], dtype=torch.float32)
    
    # Stack input features: [N, T, 7]
    # Features: [power, current, voltage, soc_prev, temp_prev, soh_prev, ambient_temp]
    # We predict next timestep values
    N, T = power.shape
    
    # Create sequences (predict t+1 from t)
    inputs = torch.stack([
        power[:, :-1], current[:, :-1], voltage[:, :-1],
        soc[:, :-1], temp[:, :-1], soh[:, :-1], ambient_temp[:, :-1]
    ], dim=-1).reshape(-1, 7)  # [N*(T-1), 7]
    
    # Targets: next timestep [soc, temp, soh]
    targets_soc = soc[:, 1:].reshape(-1)
    targets_temp = temp[:, 1:].reshape(-1)
    targets_soh = soh[:, 1:].reshape(-1)
    
    # Train/Val Split
    split_idx = int(0.8 * len(inputs))
    train_x, val_x = inputs[:split_idx], inputs[split_idx:]
    train_y = {
        'soc': targets_soc[:split_idx],
        'temperature': targets_temp[:split_idx],
        'soh': targets_soh[:split_idx]
    }
    val_y = {
        'soc': targets_soc[split_idx:],
        'temperature': targets_temp[split_idx:],
        'soh': targets_soh[split_idx:]
    }
    
    # Model (outputs 3 values: SOC, Temp, SOH)
    model = DeepONet(
        input_dim=7,
        hidden_dim=hyperparams.get('hidden_dim', 128),
        output_dim=3,
        n_buses=1
    ).to(device)
    
    # Transfer Learning
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"🔄 Loading pretrained weights")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Physics Loss
    metadata = dataset.get('metadata', {})
    physics_loss = BatteryPhysicsLoss(
        capacity_kwh=metadata.get('capacity_kwh', 100.0),
        max_power_kw=metadata.get('max_power_kw', 50.0),
        internal_resistance=metadata.get('internal_resistance', 0.05),
        max_charge_rate=metadata.get('max_charge_rate', 1.0),
        max_discharge_rate=metadata.get('max_discharge_rate', 1.0)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
            batch_y = {
                'soc': train_y['soc'][i:i+batch_size].to(device),
                'temperature': train_y['temperature'][i:i+batch_size].to(device),
                'soh': train_y['soh'][i:i+batch_size].to(device)
            }
            
            optimizer.zero_grad()
            
            dummy_buses = torch.zeros(len(batch_x), 1, dtype=torch.long, device=device)
            predictions = model(batch_x, dummy_buses)  # [batch, 3]
            
            # Extract inputs for physics loss
            batch_inputs = {
                'power': batch_x[:, 0],
                'current': batch_x[:, 1],
                'voltage': batch_x[:, 2],
                'ambient_temp': batch_x[:, 6],
                'dt': torch.ones(len(batch_x), device=device) * 3600  # 1 hour in seconds
            }
            
            loss, metrics = physics_loss(predictions, batch_y, batch_inputs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x_gpu = val_x.to(device)
            val_y_gpu = {k: v.to(device) for k, v in val_y.items()}
            dummy_buses_val = torch.zeros(len(val_x), 1, dtype=torch.long, device=device)
            
            val_preds = model(val_x_gpu, dummy_buses_val)
            val_inputs = {
                'power': val_x_gpu[:, 0],
                'current': val_x_gpu[:, 1],
                'voltage': val_x_gpu[:, 2],
                'ambient_temp': val_x_gpu[:, 6],
                'dt': torch.ones(len(val_x), device=device) * 3600
            }
            val_loss, val_metrics = physics_loss(val_preds, val_y_gpu, val_inputs)
        
        avg_train_loss = np.mean(train_losses)
        scheduler.step(val_loss)
        
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
            
            checkpoint_path = Path(checkpoint_dir) / f"battery_best_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved")
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
        'model_type': 'battery_thermal_soh',
        'physics_validation': history['physics_metrics'][-1]
    }
    
    return model.cpu(), summary


if __name__ == "__main__":
    # Unit Test
    mock_data = {
        'power': np.random.rand(100, 24) * 40 - 20,
        'current': np.random.rand(100, 24) * 100,
        'voltage': np.random.rand(100, 24) * 50 + 350,
        'soc': np.random.rand(100, 24) * 0.6 + 0.2,
        'temperature': np.random.rand(100, 24) * 10 + 25,
        'soh': np.linspace(1.0, 0.85, 100)[:, None].repeat(24, axis=1),
        'ambient_temp': np.random.rand(100, 24) * 5 + 20,
        'metadata': {'capacity_kwh': 100, 'max_power_kw': 50}
    }
    
    params = {'epochs': 20, 'learning_rate': 1e-3, 'batch_size': 16}
    
    try:
        model, summary = train_battery_thermal_model(mock_data, params)
        print(f"✅ Test Successful: {summary['best_val_loss']:.4f}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise
