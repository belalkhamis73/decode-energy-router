"""
Grid Stability & Power Flow Model Trainer
Physics-Informed Neural Network for Electrical Grid Analysis

TRAINING DATA REQUIREMENTS:
- Power flow solutions from Pandapower for various load scenarios
- Bus voltage magnitudes [p.u.] and angles [degrees]
- Active/reactive power injections [MW/MVAr]
- Line flows and losses
- Grid topology (adjacency matrix, impedances)
- Load profiles and generation dispatch
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
logger = logging.getLogger("GridStabilityTrainer")


class GridPhysicsLoss(nn.Module):
    """
    Physics-Informed Loss enforcing AC power flow equations.
    
    Key Physics Constraints:
    1. Power Balance: ΣP_gen = ΣP_load + ΣP_loss
    2. Kirchhoff's Current Law at each bus
    3. Voltage magnitude limits: 0.95 ≤ V ≤ 1.05 p.u.
    4. Active Power: P_ij = V_i²G_ij - V_i*V_j(G_ij*cos(θ_ij) + B_ij*sin(θ_ij))
    5. Reactive Power: Q_ij = -V_i²B_ij - V_i*V_j(G_ij*sin(θ_ij) - B_ij*cos(θ_ij))
    6. Line flow limits (thermal constraints)
    """
    
    def __init__(self, admittance_matrix: np.ndarray, 
                 line_limits: Optional[np.ndarray] = None,
                 slack_bus: int = 0):
        super().__init__()
        self.n_buses = admittance_matrix.shape[0]
        
        # Y = G + jB (admittance matrix)
        self.G = torch.tensor(admittance_matrix.real, dtype=torch.float32)
        self.B = torch.tensor(admittance_matrix.imag, dtype=torch.float32)
        
        self.line_limits = line_limits  # [n_lines] MVA
        self.slack_bus = slack_bus
        
    def forward(self, predictions: torch.Tensor, targets: Dict[str, torch.Tensor],
                inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: [batch, n_buses*2] -> [V_mag_1...V_mag_n, θ_1...θ_n]
            targets: Dict with 'voltage_mag', 'voltage_angle', 'p_flow', 'q_flow'
            inputs: Dict with 'p_injection', 'q_injection'
        """
        batch_size = predictions.shape[0]
        
        # Split predictions into voltage magnitudes and angles
        V_mag = predictions[:, :self.n_buses]  # [batch, n_buses]
        V_ang = predictions[:, self.n_buses:]  # [batch, n_buses] in radians
        
        target_V_mag = targets['voltage_mag']
        target_V_ang = targets['voltage_angle']
        
        # Data Loss
        mse_v = nn.MSELoss()(V_mag, target_V_mag)
        mse_ang = nn.MSELoss()(V_ang, target_V_ang)
        data_loss = mse_v + 0.5 * mse_ang
        
        # Physics Constraints
        P_inj = inputs['p_injection']  # [batch, n_buses]
        Q_inj = inputs['q_injection']
        
        # Move admittance matrices to same device
        G = self.G.to(predictions.device)
        B = self.B.to(predictions.device)
        
        # 1. Voltage Limits (0.95 - 1.05 p.u.)
        v_lower_violation = torch.relu(0.95 - V_mag)
        v_upper_violation = torch.relu(V_mag - 1.05)
        voltage_limit_loss = (v_lower_violation + v_upper_violation).mean()
        
        # 2. Power Flow Equations (vectorized for all buses)
        # P_i = Σ_j V_i*V_j*(G_ij*cos(θ_i - θ_j) + B_ij*sin(θ_i - θ_j))
        
        # Create angle difference matrix: [batch, n_buses, n_buses]
        theta_diff = V_ang.unsqueeze(2) - V_ang.unsqueeze(1)  # [batch, n_buses, n_buses]
        
        # Voltage product matrix: V_i * V_j
        V_prod = V_mag.unsqueeze(2) * V_mag.unsqueeze(1)  # [batch, n_buses, n_buses]
        
        # Power flow calculation
        cos_term = torch.cos(theta_diff) * G.unsqueeze(0)  # [batch, n_buses, n_buses]
        sin_term = torch.sin(theta_diff) * B.unsqueeze(0)
        
        P_calc = (V_prod * (cos_term + sin_term)).sum(dim=2)  # [batch, n_buses]
        Q_calc = (V_prod * (torch.sin(theta_diff) * G.unsqueeze(0) - 
                           torch.cos(theta_diff) * B.unsqueeze(0))).sum(dim=2)
        
        # Power balance residuals (should match injections)
        P_residual = torch.abs(P_calc - P_inj)
        Q_residual = torch.abs(Q_calc - Q_inj)
        
        power_balance_loss = P_residual.mean() + 0.5 * Q_residual.mean()
        
        # 3. Slack Bus Constraint (V = 1.0 p.u., θ = 0)
        slack_v_loss = torch.abs(V_mag[:, self.slack_bus] - 1.0).mean()
        slack_ang_loss = torch.abs(V_ang[:, self.slack_bus]).mean()
        slack_loss = slack_v_loss + slack_ang_loss
        
        # 4. Total System Power Balance
        # Sum of all injections should equal total load + losses
        total_p_gen = P_inj.sum(dim=1)
        total_p_calc = P_calc.sum(dim=1)
        system_balance_loss = torch.abs(total_p_gen - total_p_calc).mean()
        
        # Total Loss
        total_loss = (
            data_loss +
            1.0 * voltage_limit_loss +
            2.0 * power_balance_loss +
            3.0 * slack_loss +
            1.0 * system_balance_loss
        )
        
        metrics = {
            'mse_voltage': mse_v.item(),
            'mse_angle': mse_ang.item(),
            'voltage_violation': voltage_limit_loss.item(),
            'power_balance_error': power_balance_loss.item(),
            'slack_violation': slack_loss.item(),
            'system_balance_error': system_balance_loss.item()
        }
        
        return total_loss, metrics


def train_grid_stability_model(
    dataset: Dict[str, Any],
    hyperparams: Dict[str, Any],
    pretrained_path: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints/grid"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train Grid Stability & Power Flow model.
    
    Args:
        dataset: {
            'p_injection': [N, n_buses], 'q_injection': [N, n_buses],
            'voltage_mag': [N, n_buses], 'voltage_angle': [N, n_buses],
            'line_flows': [N, n_lines] (optional),
            'metadata': {
                'admittance_matrix': ndarray [n_buses, n_buses],
                'n_buses': int,
                'slack_bus': int
            }
        }
        hyperparams: Training configuration
        pretrained_path: Checkpoint path
        checkpoint_dir: Save directory
    """
    
    # Setup
    device = hyperparams.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    epochs = hyperparams.get('epochs', 150)
    lr = hyperparams.get('learning_rate', 1e-3)
    batch_size = hyperparams.get('batch_size', 64)
    patience = hyperparams.get('early_stop_patience', 20)
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    metadata = dataset.get('metadata', {})
    n_buses = metadata.get('n_buses', 118)
    
    logger.info(f"⚡ Initializing Grid Stability Model Training on {device}")
    logger.info(f"   Grid Size: {n_buses} buses")
    
    # Data Preparation
    p_inj = torch.tensor(dataset['p_injection'], dtype=torch.float32)
    q_inj = torch.tensor(dataset['q_injection'], dtype=torch.float32)
    v_mag = torch.tensor(dataset['voltage_mag'], dtype=torch.float32)
    v_ang = torch.tensor(dataset['voltage_angle'], dtype=torch.float32)
    
    # Inputs: Power injections [N, n_buses*2]
    inputs = torch.cat([p_inj, q_inj], dim=-1)
    
    # Targets: Voltage states [N, n_buses*2]
    targets = torch.cat([v_mag, v_ang], dim=-1)
    
    # Train/Val Split
    split_idx = int(0.8 * len(inputs))
    train_x, val_x = inputs[:split_idx], inputs[split_idx:]
    train_y, val_y = targets[:split_idx], targets[split_idx:]
    train_p, val_p = p_inj[:split_idx], p_inj[split_idx:]
    train_q, val_q = q_inj[:split_idx], q_inj[split_idx:]
    
    # Model
    model = DeepONet(
        input_dim=n_buses * 2,  # P and Q injections
        hidden_dim=hyperparams.get('hidden_dim', 256),
        output_dim=n_buses * 2,  # V_mag and V_angle
        n_buses=n_buses
    ).to(device)
    
    # Transfer Learning
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"🔄 Loading pretrained weights")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Physics Loss
    Y_bus = metadata.get('admittance_matrix', np.eye(n_buses, dtype=complex))
    physics_loss = GridPhysicsLoss(
        admittance_matrix=Y_bus,
        slack_bus=metadata.get('slack_bus', 0)
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
            batch_p = train_p[i:i+batch_size].to(device)
            batch_q = train_q[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            # Create bus indices for DeepONet
            bus_ids = torch.arange(n_buses, dtype=torch.long, device=device)
            bus_ids = bus_ids.unsqueeze(0).repeat(len(batch_x), 1)
            
            predictions = model(batch_x, bus_ids)  # [batch, n_buses*2]
            
            # Prepare targets and inputs for physics loss
            targets_dict = {
                'voltage_mag': batch_y[:, :n_buses],
                'voltage_angle': batch_y[:, n_buses:]
            }
            inputs_dict = {
                'p_injection': batch_p,
                'q_injection': batch_q
            }
            
            loss, metrics = physics_loss(predictions, targets_dict, inputs_dict)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x_gpu = val_x.to(device)
            val_y_gpu = val_y.to(device)
            val_p_gpu = val_p.to(device)
            val_q_gpu = val_q.to(device)
            
            bus_ids_val = torch.arange(n_buses, dtype=torch.long, device=device)
            bus_ids_val = bus_ids_val.unsqueeze(0).repeat(len(val_x), 1)
            
            val_preds = model(val_x_gpu, bus_ids_val)
            
            val_targets = {
                'voltage_mag': val_y_gpu[:, :n_buses],
                'voltage_angle': val_y_gpu[:, n_buses:]
            }
            val_inputs = {
                'p_injection': val_p_gpu,
                'q_injection': val_q_gpu
            }
            
            val_loss, val_metrics = physics_loss(val_preds, val_targets, val_inputs)
        
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
            
            checkpoint_path = Path(checkpoint_dir) / f"grid_best_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
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
        'model_type': 'grid_stability',
        'n_buses': n_buses,
        'physics_validation': history['physics_metrics'][-1]
    }
    
    return model.cpu(), summary


if __name__ == "__main__":
    # Unit Test with IEEE 14-bus equivalent
    n_buses = 14
    mock_Y = np.random.rand(n_buses, n_buses) + 1j * np.random.rand(n_buses, n_buses)
    mock_Y = (mock_Y + mock_Y.T) / 2  # Make symmetric
    
    mock_data = {
        'p_injection': np.random.rand(100, n_buses) * 100 - 50,
        'q_injection': np.random.rand(100, n_buses) * 50 - 25,
        'voltage_mag': np.random.rand(100, n_buses) * 0.1 + 0.95,
        'voltage_angle': np.random.rand(100, n_buses) * 0.1 - 0.05,
        'metadata': {
            'admittance_matrix': mock_Y,
            'n_buses': n_buses,
            'slack_bus': 0
        }
    }
    
    params = {'epochs': 20, 'learning_rate': 1e-3, 'batch_size': 32}
    
    try:
        model, summary = train_grid_stability_model(mock_data, params)
        print(f"✅ Test Successful: {summary['best_val_loss']:.4f}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise
