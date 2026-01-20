"""
Load Forecasting Model Trainer
Physics-Informed Neural Network for Electrical Load Prediction

TRAINING DATA REQUIREMENTS:
- Historical demand data (hourly) [MW]
- Weather features: temperature, humidity, wind speed
- Calendar features: hour, day of week, month, holidays
- Lagged demand values (past 24-168 hours)
- Special events flags
- Day-ahead temperature forecasts
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
logger = logging.getLogger("LoadForecastTrainer")


class LoadForecastPhysicsLoss(nn.Module):
    """
    Physics-Informed Loss for load forecasting.
    
    Key Constraints:
    1. Non-negativity: Load ≥ 0
    2. Temporal smoothness: |L(t+1) - L(t)| < ΔL_max (no sudden jumps)
    3. Daily periodicity preservation
    4. Temperature correlation: Load increases with extreme temps
    5. Peak load bounds: L ≤ L_historical_max * 1.2
    6. Minimum base load: L ≥ L_base_min
    """
    
    def __init__(self, historical_max_load: float, base_load: float,
                 max_ramp_rate: float = 100.0):  # MW/hour
        super().__init__()
        self.L_max = historical_max_load * 1.2  # Allow 20% headroom
        self.L_base = base_load * 0.8  # 80% of typical base
        self.max_ramp = max_ramp_rate
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: [batch, horizon] predicted load in MW
            targets: [batch, horizon] actual load measurements
            inputs: Dict with 'temperature', 'hour', 'prev_load'
        """
        # Data Loss
        mse_loss = nn.MSELoss()(predictions, targets)
        mae_loss = nn.L1Loss()(predictions, targets)
        data_loss = 0.7 * mse_loss + 0.3 * mae_loss
        
        # Physics Constraints
        
        # 1. Non-negativity
        negativity_loss = torch.relu(-predictions).mean()
        
        # 2. Peak Load Constraint
        peak_violation = torch.relu(predictions - self.L_max)
        peak_loss = peak_violation.mean()
        
        # 3. Base Load Constraint
        base_violation = torch.relu(self.L_base - predictions)
        base_loss = base_violation.mean()
        
        # 4. Temporal Smoothness (ramp rate constraint)
        if predictions.shape[1] > 1:
            load_diff = predictions[:, 1:] - predictions[:, :-1]
            ramp_violation = torch.relu(torch.abs(load_diff) - self.max_ramp)
            ramp_loss = ramp_violation.mean()
        else:
            ramp_loss = torch.tensor(0.0, device=predictions.device)
        
        # 5. Temperature Correlation (simplified)
        # High/low temps should correlate with higher load
        if 'temperature' in inputs:
            temp = inputs['temperature']
            temp_deviation = torch.abs(temp - 20.0)  # Comfort zone at 20°C
            
            # Expect higher load when temp deviates from comfort
            # This is a soft constraint
            temp_load_correlation = -torch.corrcoef(
                torch.stack([temp_deviation.flatten(), predictions.flatten()])
            )[0, 1]
            
            # Penalize negative correlation (should be positive)
            temp_loss = torch.relu(-temp_load_correlation)
        else:
            temp_loss = torch.tensor(0.0, device=predictions.device)
        
        # 6. Daily Periodicity (24-hour pattern similarity)
        if predictions.shape[1] >= 24:
            # Compare 24h segments - they should have similar patterns
            segment1 = predictions[:, :24]
            segment2 = predictions[:, -24:] if predictions.shape[1] >= 48 else segment1
            
            # Normalized pattern similarity
            pattern_diff = torch.abs(
                segment1 / (segment1.mean(dim=1, keepdim=True) + 1e-6) -
                segment2 / (segment2.mean(dim=1, keepdim=True) + 1e-6)
            )
            periodicity_loss = pattern_diff.mean()
        else:
            periodicity_loss = torch.tensor(0.0, device=predictions.device)
        
        # Total Loss
        total_loss = (
            data_loss +
            5.0 * negativity_loss +
            2.0 * peak_loss +
            1.0 * base_loss +
            1.5 * ramp_loss +
            0.3 * temp_loss +
            0.5 * periodicity_loss
        )
        
        metrics = {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'negativity': negativity_loss.item(),
            'peak_violation': peak_loss.item(),
            'base_violation': base_loss.item(),
            'ramp_violation': ramp_loss.item() if isinstance(ramp_loss, torch.Tensor) else 0.0,
            'temp_correlation_penalty': temp_loss.item() if isinstance(temp_loss, torch.Tensor) else 0.0,
            'periodicity_error': periodicity_loss.item() if isinstance(periodicity_loss, torch.Tensor) else 0.0
        }
        
        return total_loss, metrics


def train_load_forecast_model(
    dataset: Dict[str, Any],
    hyperparams: Dict[str, Any],
    pretrained_path: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints/load_forecast"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train Load Forecasting model.
    
    Args:
        dataset: {
            'load': [N, T+H] historical + future load (MW),
            'temperature': [N, T+H],
            'humidity': [N, T+H],
            'hour': [N, T+H] (0-23),
            'day_of_week': [N, T+H] (0-6),
            'month': [N, T+H] (1-12),
            'is_holiday': [N, T+H] (binary),
            'metadata': {
                'historical_max_load': float,
                'base_load': float,
                'forecast_horizon': int (default 24)
            }
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
    
    metadata = dataset.get('metadata', {})
    forecast_horizon = metadata.get('forecast_horizon', 24)
    lookback_window = hyperparams.get('lookback_window', 168)  # 1 week
    
    logger.info(f"📊 Initializing Load Forecast Model Training on {device}")
    logger.info(f"   Forecast Horizon: {forecast_horizon}h | Lookback: {lookback_window}h")
    
    # Data Preparation
    load = torch.tensor(dataset['load'], dtype=torch.float32)
    temp = torch.tensor(dataset['temperature'], dtype=torch.float32)
    humidity = torch.tensor(dataset['humidity'], dtype=torch.float32)
    hour = torch.tensor(dataset['hour'], dtype=torch.float32)
    day_of_week = torch.tensor(dataset['day_of_week'], dtype=torch.float32)
    month = torch.tensor(dataset['month'], dtype=torch.float32)
    is_holiday = torch.tensor(dataset['is_holiday'], dtype=torch.float32)
    
    # Create sequences: Use past 'lookback_window' to predict next 'forecast_horizon'
    N, T = load.shape
    
    sequences_x = []
    sequences_y = []
    
    for i in range(N):
        for t in range(lookback_window, T - forecast_horizon):
            # Input features: [lookback_window, 7]
            # Features: load, temp, humidity, hour, day_of_week, month, is_holiday
            x_seq = torch.stack([
                load[i, t-lookback_window:t],
                temp[i, t-lookback_window:t],
                humidity[i, t-lookback_window:t],
                hour[i, t-lookback_window:t],
                day_of_week[i, t-lookback_window:t],
                month[i, t-lookback_window:t],
                is_holiday[i, t-lookback_window:t]
            ], dim=-1)  # [lookback_window, 7]
            
            # Target: next 'forecast_horizon' load values
            y_seq = load[i, t:t+forecast_horizon]  # [forecast_horizon]
            
            sequences_x.append(x_seq)
            sequences_y.append(y_seq)
    
    # Stack all sequences
    inputs = torch.stack(sequences_x)  # [num_sequences, lookback_window, 7]
    targets = torch.stack(sequences_y)  # [num_sequences, forecast_horizon]
    
    # Flatten temporal dimension for model input
    inputs = inputs.reshape(len(inputs), -1)  # [num_sequences, lookback_window*7]
    
    logger.info(f"   Total sequences: {len(inputs)}")
    
    # Train/Val Split
    split_idx = int(0.8 * len(inputs))
    train_x, val_x = inputs[:split_idx], inputs[split_idx:]
    train_y, val_y = targets[:split_idx], targets[split_idx:]
    
    # Extract temperature for physics loss
    temp_train = temp[:split_idx].reshape(-1)[:len(train_y)]
    temp_val = temp[split_idx:].reshape(-1)[:len(val_y)]
    
    # Model
    model = DeepONet(
        input_dim=lookback_window * 7,
        hidden_dim=hyperparams.get('hidden_dim', 256),
        output_dim=forecast_horizon,
        n_buses=1
    ).to(device)
    
    # Transfer Learning
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"🔄 Loading pretrained weights")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Physics Loss
    physics_loss = LoadForecastPhysicsLoss(
        historical_max_load=metadata.get('historical_max_load', 1000.0),
        base_load=metadata.get('base_load', 200.0),
        max_ramp_rate=metadata.get('max_ramp_rate', 100.0)
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
            batch_y = train_y[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            dummy_buses = torch.zeros(len(batch_x), 1, dtype=torch.long, device=device)
            predictions = model(batch_x, dummy_buses).squeeze(-1)  # [batch, forecast_horizon]
            
            # Extract temperature from input sequence (use mean of lookback)
            batch_temp = batch_x[:, :lookback_window].mean(dim=1)  # Simplified
            
            inputs_dict = {
                'temperature': batch_temp,
                'hour': None,
                'prev_load': batch_x[:, :lookback_window]
            }
            
            loss, metrics = physics_loss(predictions, batch_y, inputs_dict)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x_gpu = val_x.to(device)
            val_y_gpu = val_y.to(device)
            dummy_buses_val = torch.zeros(len(val_x), 1, dtype=torch.long, device=device)
            
            val_preds = model(val_x_gpu, dummy_buses_val).squeeze(-1)
            
            val_temp = val_x_gpu[:, :lookback_window].mean(dim=1)
            val_inputs = {
                'temperature': val_temp,
                'hour': None,
                'prev_load': val_x_gpu[:, :lookback_window]
            }
            
            val_loss, val_metrics = physics_loss(val_preds, val_y_gpu, val_inputs)
        
        avg_train_loss = np.mean(train_losses)
        scheduler.step(val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['physics_metrics'].append(val_metrics)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss.item():.4f} | MAE: {val_metrics['mae']:.2f} MW")
        
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
            
            checkpoint_path = Path(checkpoint_dir) / f"load_forecast_best_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
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
        'model_type': 'load_forecast',
        'forecast_horizon': forecast_horizon,
        'physics_validation': history['physics_metrics'][-1]
    }
    
    return model.cpu(), summary


if __name__ == "__main__":
    # Unit Test
    N_samples = 200
    T_total = 240  # 10 days hourly
    
    mock_data = {
        'load': np.random.rand(N_samples, T_total) * 500 + 300,
        'temperature': np.random.rand(N_samples, T_total) * 20 + 10,
        'humidity': np.random.rand(N_samples, T_total) * 40 + 40,
        'hour': np.tile(np.arange(24), T_total // 24 + 1)[:T_total][None, :].repeat(N_samples, axis=0),
        'day_of_week': np.tile(np.arange(7), T_total // 168 + 1)[:T_total][None, :].repeat(N_samples, axis=0),
        'month': np.ones((N_samples, T_total)) * 6,
        'is_holiday': np.zeros((N_samples, T_total)),
        'metadata': {
            'historical_max_load': 800.0,
            'base_load': 250.0,
            'forecast_horizon': 24
        }
    }
    
    params = {'epochs': 20, 'learning_rate': 1e-3, 'batch_size': 16, 'lookback_window': 48}
    
    try:
        model, summary = train_load_forecast_model(mock_data, params)
        print(f"✅ Test Successful: Val Loss={summary['best_val_loss']:.4f}, MAE={summary['physics_validation']['mae']:.2f} MW")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise
