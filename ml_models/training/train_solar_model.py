"""
Solar PV Generation Model Trainer
Physics-Informed Neural Network for Solar Power Prediction

TRAINING DATA REQUIREMENTS:
- 1 year hourly GHI (Global Horizontal Irradiance) [W/m²]
- DNI (Direct Normal Irradiance) [W/m²]
- DHI (Diffuse Horizontal Irradiance) [W/m²]
- Actual PV output measurements [kW]
- Temperature [°C]
- Module temperature [°C] (optional)
- Array configuration metadata
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

from ml_models.architectures.deeponet import DeepONet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolarModelTrainer")


class SolarPhysicsLoss(nn.Module):
    """
    Physics-Informed Loss enforcing solar PV equations.
    
    Key Physics Constraints:
    1. P_out ≤ P_rated (Physical capacity constraint)
    2. P_out = η * GHI * A * PR (Simplified PV equation)
    3. Efficiency degrades with temperature: η(T) = η_ref * (1 - β(T - T_ref))
    4. Zero output when GHI < threshold (nighttime constraint)
    """
    
    def __init__(self, rated_power: float, array_area: float, 
                 performance_ratio: float = 0.75, temp_coeff: float = -0.004):
        super().__init__()
        self.P_rated = rated_power  # kW
        self.A = array_area  # m²
        self.PR = performance_ratio
        self.beta = temp_coeff  # per °C
        self.T_ref = 25.0  # °C
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: [batch, 1] predicted power output
            targets: [batch, 1] actual measurements
            inputs: Dict with 'ghi', 'temperature', 'dhi', 'dni'
        """
        # Data Loss
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # Physics Loss Components
        ghi = inputs['ghi'].unsqueeze(-1)  # [batch, 1]
        temp = inputs['temperature'].unsqueeze(-1)
        
        # 1. Capacity Constraint: P ≤ P_rated
        capacity_violation = torch.relu(predictions - self.P_rated)
        capacity_loss = capacity_violation.mean()
        
        # 2. Temperature-corrected efficiency
        eta_temp = 1.0 + self.beta * (temp - self.T_ref)
        theoretical_power = eta_temp * ghi * self.A * self.PR / 1000.0  # Convert to kW
        physics_loss = nn.MSELoss()(predictions, theoretical_power.clamp(0, self.P_rated))
        
        # 3. Nighttime Constraint: P ≈ 0 when GHI < 10 W/m²
        nighttime_mask = (ghi < 10.0).float()
        nighttime_loss = (predictions * nighttime_mask).abs().mean()
        
        # 4. Non-negativity
        negativity_loss = torch.relu(-predictions).mean()
        
        # Weighted Total Loss
        total_loss = (
            mse_loss + 
            0.5 * physics_loss + 
            2.0 * capacity_loss + 
            1.0 * nighttime_loss + 
            5.0 * negativity_loss
        )
        
        metrics = {
            'mse': mse_loss.item(),
            'physics': physics_loss.item(),
            'capacity_violation': capacity_loss.item(),
            'nighttime_violation': nighttime_loss.item(),
            'negativity': negativity_loss.item()
        }
        
        return total_loss, metrics


def train_solar_model(
    dataset: Dict[str, Any],
    hyperparams: Dict[str, Any],
    pretrained_path: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints/solar"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train Solar PV prediction model with physics constraints.
    
    Args:
        dataset: {
            'ghi': [N, 24], 'dni': [N, 24], 'dhi': [N, 24],
            'temperature': [N, 24], 'actual_output': [N, 24],
            'metadata': {'rated_power': float, 'array_area': float}
        }
        hyperparams: {
            'epochs': int, 'learning_rate': float, 'batch_size': int,
            'device': str, 'early_stop_patience': int
        }
        pretrained_path: Path to checkpoint for transfer learning
        checkpoint_dir: Directory for saving versioned checkpoints
        
    Returns:
        model: Trained PyTorch model
        training_summary: Metrics and validation results
    """
    
    # Setup
    device = hyperparams.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    epochs = hyperparams.get('epochs', 100)
    lr = hyperparams.get('learning_rate', 1e-3)
    batch_size = hyperparams.get('batch_size', 32)
    patience = hyperparams.get('early_stop_patience', 15)
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"☀️ Initializing Solar Model Training on {device}")
    
    # Data Preparation
    ghi = torch.tensor(dataset['ghi'], dtype=torch.float32)
    dni = torch.tensor(dataset['dni'], dtype=torch.float32)
    dhi = torch.tensor(dataset['dhi'], dtype=torch.float32)
    temp = torch.tensor(dataset['temperature'], dtype=torch.float32)
    targets = torch.tensor(dataset['actual_output'], dtype=torch.float32)
    
    # Stack features: [N, 24, 4]
    N = ghi.shape[0]
    features = torch.stack([ghi, dni, dhi, temp], dim=-1).reshape(-1, 4)  # [N*24, 4]
    targets = targets.reshape(-1, 1)  # [N*24, 1]
    
    # Train/Val Split (80/20)
    split_idx = int(0.8 * len(features))
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = targets[:split_idx], targets[split_idx:]
    
    # Model Initialization
    model = DeepONet(
        input_dim=4,
        hidden_dim=hyperparams.get('hidden_dim', 128),
        output_dim=1,
        n_buses=1  # Single output (power)
    ).to(device)
    
    # Transfer Learning
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"🔄 Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Physics-Informed Loss
    metadata = dataset.get('metadata', {})
    physics_loss = SolarPhysicsLoss(
        rated_power=metadata.get('rated_power', 100.0),  # kW
        array_area=metadata.get('array_area', 500.0),  # m²
        performance_ratio=metadata.get('performance_ratio', 0.75)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'physics_metrics': []}
    
    logger.info(f"🚀 Training for {epochs} epochs | Batch Size: {batch_size}")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Mini-batch training
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size].to(device)
            batch_y = train_y[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (DeepONet expects [batch, features] and dummy bus_ids)
            dummy_buses = torch.zeros(len(batch_x), 1, dtype=torch.long, device=device)
            predictions = model(batch_x, dummy_buses)  # [batch, 1]
            
            # Physics-informed loss
            inputs_dict = {
                'ghi': batch_x[:, 0],
                'dni': batch_x[:, 1],
                'dhi': batch_x[:, 2],
                'temperature': batch_x[:, 3]
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
            
            val_preds = model(val_x_gpu, dummy_buses_val)
            val_inputs = {
                'ghi': val_x_gpu[:, 0],
                'dni': val_x_gpu[:, 1],
                'dhi': val_x_gpu[:, 2],
                'temperature': val_x_gpu[:, 3]
            }
            val_loss, val_metrics = physics_loss(val_preds, val_y_gpu, val_inputs)
        
        avg_train_loss = np.mean(train_losses)
        scheduler.step(val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['physics_metrics'].append(val_metrics)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss.item():.4f} | Physics: {val_metrics['physics']:.4f}")
        
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
            
            checkpoint_path = Path(checkpoint_dir) / f"solar_best_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"⏸️ Early stopping at epoch {epoch}")
            break
    
    logger.info(f"✅ Training Complete | Best Val Loss: {best_val_loss:.4f}")
    
    # Training Summary
    summary = {
        'final_train_loss': history['train_loss'][-1],
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'history': history,
        'model_type': 'solar_pv',
        'physics_validation': history['physics_metrics'][-1]
    }
    
    return model.cpu(), summary


# Online Learning (Fine-tuning from live data)
def fine_tune_solar_model(
    model: nn.Module,
    new_data: Dict[str, np.ndarray],
    n_iterations: int = 10,
    lr: float = 1e-4
) -> nn.Module:
    """
    Fine-tune existing model with recent operational data.
    
    Args:
        model: Pretrained solar model
        new_data: {'ghi', 'dni', 'dhi', 'temperature', 'actual_output'}
        n_iterations: Number of fine-tuning steps
        lr: Learning rate (should be lower than initial training)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare data
    features = torch.tensor(np.stack([
        new_data['ghi'], new_data['dni'], 
        new_data['dhi'], new_data['temperature']
    ], axis=-1), dtype=torch.float32).to(device)
    
    targets = torch.tensor(new_data['actual_output'], dtype=torch.float32).unsqueeze(-1).to(device)
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        dummy_buses = torch.zeros(len(features), 1, dtype=torch.long, device=device)
        predictions = model(features, dummy_buses)
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            logger.info(f"Fine-tune iteration {i}: Loss = {loss.item():.4f}")
    
    return model.cpu()


if __name__ == "__main__":
    # Unit Test
    mock_data = {
        'ghi': np.random.rand(100, 24) * 800,
        'dni': np.random.rand(100, 24) * 600,
        'dhi': np.random.rand(100, 24) * 200,
        'temperature': np.random.rand(100, 24) * 15 + 20,
        'actual_output': np.random.rand(100, 24) * 90,
        'metadata': {'rated_power': 100, 'array_area': 500}
    }
    
    params = {'epochs': 20, 'learning_rate': 1e-3, 'batch_size': 16}
    
    try:
        model, summary = train_solar_model(mock_data, params)
        print(f"✅ Test Successful: {summary['best_val_loss']:.4f}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise
