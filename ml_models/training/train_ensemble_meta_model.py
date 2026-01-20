"""
Ensemble Meta-Model Trainer
Combines predictions from all specialized models (Solar, Wind, Battery, Grid, Load)
with weighted fusion and uncertainty quantification.

TRAINING APPROACH:
- Stacked Ensemble: Uses outputs from base models as features
- Learns optimal weighting based on historical performance
- Incorporates cross-model consistency checks
- Provides confidence intervals and uncertainty estimates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnsembleMetaTrainer")


class MetaLearner(nn.Module):
    """
    Meta-learner that combines predictions from multiple specialized models.
    
    Architecture:
    - Takes base model predictions as input
    - Learns adaptive weights based on context
    - Outputs final prediction + uncertainty estimate
    """
    
    def __init__(self, n_base_models: int, context_dim: int = 10,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.n_models = n_base_models
        
        # Context encoder (learns when to trust which model)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for adaptive weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + n_base_models, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_base_models),
            nn.Softmax(dim=-1)
        )
        
        # Uncertainty estimator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim + n_base_models, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
    def forward(self, base_predictions: torch.Tensor, 
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base_predictions: [batch, n_models] predictions from base models
            context: [batch, context_dim] contextual features
            
        Returns:
            final_prediction: [batch, 1] weighted ensemble prediction
            uncertainty: [batch, 1] prediction uncertainty (std dev)
        """
        # Encode context
        ctx_encoded = self.context_encoder(context)
        
        # Compute adaptive weights
        attn_input = torch.cat([ctx_encoded, base_predictions], dim=-1)
        weights = self.attention(attn_input)  # [batch, n_models]
        
        # Weighted combination
        final_pred = (base_predictions * weights).sum(dim=-1, keepdim=True)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(attn_input)
        
        return final_pred, uncertainty


class EnsemblePhysicsLoss(nn.Module):
    """
    Meta-level physics constraints ensuring ensemble consistency.
    
    Key Constraints:
    1. Cross-model consistency (predictions shouldn't wildly disagree)
    2. Uncertainty calibration (higher uncertainty when models disagree)
    3. Conservative prediction bounds
    4. Energy conservation across subsystems
    """
    
    def __init__(self, consistency_threshold: float = 0.2):
        super().__init__()
        self.consistency_threshold = consistency_threshold
        
    def forward(self, ensemble_pred: torch.Tensor, uncertainty: torch.Tensor,
                base_predictions: torch.Tensor, targets: torch.Tensor,
                subsystem_constraints: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            ensemble_pred: [batch, 1] meta-model prediction
            uncertainty: [batch, 1] predicted uncertainty
            base_predictions: [batch, n_models] individual model predictions
            targets: [batch, 1] ground truth
            subsystem_constraints: Optional physics constraints from subsystems
        """
        # 1. Prediction Accuracy Loss
        mse_loss = nn.MSELoss()(ensemble_pred, targets)
        
        # 2. Uncertainty Calibration Loss (NLL - Negative Log Likelihood)
        # Assumes Gaussian distribution
        nll_loss = 0.5 * torch.log(uncertainty + 1e-6) + \
                   0.5 * ((ensemble_pred - targets) ** 2) / (uncertainty + 1e-6)
        nll_loss = nll_loss.mean()
        
        # 3. Cross-Model Consistency
        # Variance across base model predictions
        pred_std = base_predictions.std(dim=1, keepdim=True)
        
        # Uncertainty should reflect disagreement
        uncertainty_consistency = torch.abs(uncertainty - pred_std)
        consistency_loss = uncertainty_consistency.mean()
        
        # 4. Prediction should be within base model range (ensemble bound)
        pred_min = base_predictions.min(dim=1, keepdim=True)[0]
        pred_max = base_predictions.max(dim=1, keepdim=True)[0]
        
        below_min = torch.relu(pred_min - ensemble_pred)
        above_max = torch.relu(ensemble_pred - pred_max)
        bound_violation_loss = (below_min + above_max).mean()
        
        # 5. Subsystem Energy Conservation (if provided)
        if subsystem_constraints:
            # Example: Total generation should equal total load + losses
            energy_balance_loss = subsystem_constraints.get('energy_balance_error', 0.0)
        else:
            energy_balance_loss = 0.0
        
        # Total Loss
        total_loss = (
            mse_loss +
            0.5 * nll_loss +
            0.3 * consistency_loss +
            0.2 * bound_violation_loss +
            0.1 * energy_balance_loss
        )
        
        metrics = {
            'mse': mse_loss.item(),
            'nll': nll_loss.item(),
            'consistency_error': consistency_loss.item(),
            'bound_violation': bound_violation_loss.item(),
            'mean_uncertainty': uncertainty.mean().item(),
            'prediction_std': pred_std.mean().item()
        }
        
        return total_loss, metrics


def train_ensemble_meta_model(
    dataset: Dict[str, Any],
    base_model_registry: Dict[str, str],  # {'solar': 'path/to/checkpoint', ...}
    hyperparams: Dict[str, Any],
    checkpoint_dir: str = "./checkpoints/ensemble"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train Ensemble Meta-Model.
    
    Args:
        dataset: {
            'base_predictions': {
                'solar': [N, T],
                'wind': [N, T],
                'battery_soc': [N, T],
                'grid_voltage': [N, T],
                'load_forecast': [N, T]
            },
            'context_features': [N, T, context_dim],
            'targets': [N, T] (actual measurements),
            'subsystem_data': Dict (optional physics constraints)
        }
        base_model_registry: Paths to trained base models
        hyperparams: Training configuration
        checkpoint_dir: Save directory
    """
    
    # Setup
    device = hyperparams.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    epochs = hyperparams.get('epochs', 80)
    lr = hyperparams.get('learning_rate', 1e-3)
    batch_size = hyperparams.get('batch_size', 64)
    patience = hyperparams.get('early_stop_patience', 12)
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎯 Initializing Ensemble Meta-Model Training on {device}")
    logger.info(f"   Base Models: {list(base_model_registry.keys())}")
    
    # Data Preparation
    base_pred_dict = dataset['base_predictions']
    context = torch.tensor(dataset['context_features'], dtype=torch.float32)
    targets = torch.tensor(dataset['targets'], dtype=torch.float32)
    
    # Stack base model predictions: [N, T, n_models]
    model_names = sorted(base_pred_dict.keys())
    n_models = len(model_names)
    
    base_predictions = torch.stack([
        torch.tensor(base_pred_dict[name], dtype=torch.float32)
        for name in model_names
    ], dim=-1)  # [N, T, n_models]
    
    # Reshape for training: [N*T, n_models]
    N, T, _ = base_predictions.shape
    base_predictions = base_predictions.reshape(-1, n_models)
    context = context.reshape(-1, context.shape[-1])
    targets = targets.reshape(-1, 1)
    
    logger.info(f"   Total samples: {len(targets)}")
    logger.info(f"   Context dimension: {context.shape[-1]}")
    
    # Train/Val Split
    split_idx = int(0.8 * len(base_predictions))
    train_pred, val_pred = base_predictions[:split_idx], base_predictions[split_idx:]
    train_ctx, val_ctx = context[:split_idx], context[split_idx:]
    train_y, val_y = targets[:split_idx], targets[split_idx:]
    
    # Model
    meta_model = MetaLearner(
        n_base_models=n_models,
        context_dim=context.shape[-1],
        hidden_dim=hyperparams.get('hidden_dim', 64)
    ).to(device)
    
    # Physics Loss
    physics_loss = EnsemblePhysicsLoss(
        consistency_threshold=hyperparams.get('consistency_threshold', 0.2)
    ).to(device)
    
    optimizer = optim.AdamW(meta_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 
        'physics_metrics': [],
        'model_weights': []  # Track learned weights over time
    }
    
    logger.info(f"🚀 Training for {epochs} epochs")
    
    for epoch in range(epochs):
        meta_model.train()
        train_losses = []
        
        for i in range(0, len(train_pred), batch_size):
            batch_pred = train_pred[i:i+batch_size].to(device)
            batch_ctx = train_ctx[i:i+batch_size].to(device)
            batch_y = train_y[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            ensemble_pred, uncertainty = meta_model(batch_pred, batch_ctx)
            
            # Loss computation
            loss, metrics = physics_loss(
                ensemble_pred, uncertainty, batch_pred, batch_y
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        meta_model.eval()
        with torch.no_grad():
            val_pred_gpu = val_pred.to(device)
            val_ctx_gpu = val_ctx.to(device)
            val_y_gpu = val_y.to(device)
            
            val_ensemble_pred, val_uncertainty = meta_model(val_pred_gpu, val_ctx_gpu)
            val_loss, val_metrics = physics_loss(
                val_ensemble_pred, val_uncertainty, val_pred_gpu, val_y_gpu
            )
            
            # Analyze learned weights (sample batch)
            sample_ctx = val_ctx_gpu[:100]
            sample_pred = val_pred_gpu[:100]
            ctx_encoded = meta_model.context_encoder(sample_ctx)
            attn_input = torch.cat([ctx_encoded, sample_pred], dim=-1)
            sample_weights = meta_model.attention(attn_input)
            avg_weights = sample_weights.mean(dim=0).cpu().numpy()
        
        avg_train_loss = np.mean(train_losses)
        scheduler.step(val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['physics_metrics'].append(val_metrics)
        history['model_weights'].append({
            name: float(weight) for name, weight in zip(model_names, avg_weights)
        })
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Train: {avg_train_loss:.4f} | Val: {val_loss.item():.4f}")
            logger.info(f"   Avg Weights: {', '.join([f'{name[:4]}={avg_weights[i]:.3f}' for i, name in enumerate(model_names)])}")
            logger.info(f"   Mean Uncertainty: {val_metrics['mean_uncertainty']:.4f}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': meta_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss.item(),
                'hyperparams': hyperparams,
                'base_model_registry': base_model_registry,
                'model_names': model_names,
                'avg_weights': avg_weights.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_path = Path(checkpoint_dir) / f"ensemble_best_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Save metadata
            metadata_path = Path(checkpoint_dir) / "ensemble_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_names': model_names,
                    'best_weights': {name: float(w) for name, w in zip(model_names, avg_weights)},
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"💾 Checkpoint saved")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"⏸️ Early stopping at epoch {epoch}")
            break
    
    logger.info(f"✅ Training Complete | Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"   Final Model Weights: {history['model_weights'][-1]}")
    
    summary = {
        'final_train_loss': history['train_loss'][-1],
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'history': history,
        'model_type': 'ensemble_meta',
        'base_models': model_names,
        'learned_weights': history['model_weights'][-1],
        'physics_validation': history['physics_metrics'][-1]
    }
    
    return meta_model.cpu(), summary


def inference_with_uncertainty(
    meta_model: nn.Module,
    base_predictions: Dict[str, np.ndarray],
    context: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Generate ensemble predictions with confidence intervals.
    
    Args:
        meta_model: Trained ensemble model
        base_predictions: Dict of predictions from each base model
        context: Contextual features
        confidence_level: Confidence level for intervals (default 95%)
        
    Returns:
        Dict with 'prediction', 'lower_bound', 'upper_bound', 'uncertainty'
    """
    meta_model.eval()
    
    # Prepare inputs
    model_names = sorted(base_predictions.keys())
    base_pred_tensor = torch.tensor(np.stack([
        base_predictions[name] for name in model_names
    ], axis=-1), dtype=torch.float32)
    
    context_tensor = torch.tensor(context, dtype=torch.float32)
    
    with torch.no_grad():
        prediction, uncertainty = meta_model(base_pred_tensor, context_tensor)
        
        prediction = prediction.numpy()
        uncertainty = uncertainty.numpy()
        
        # Compute confidence intervals (assuming Gaussian)
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bound = prediction - z_score * uncertainty
        upper_bound = prediction + z_score * uncertainty
    
    return {
        'prediction': prediction,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'uncertainty': uncertainty
    }


if __name__ == "__main__":
    # Unit Test
    n_samples = 500
    n_timesteps = 24
    n_models = 5
    context_dim = 10
    
    mock_data = {
        'base_predictions': {
            'solar': np.random.rand(n_samples, n_timesteps) * 100,
            'wind': np.random.rand(n_samples, n_timesteps) * 200,
            'battery_soc': np.random.rand(n_samples, n_timesteps),
            'grid_voltage': np.random.rand(n_samples, n_timesteps) * 0.1 + 0.95,
            'load_forecast': np.random.rand(n_samples, n_timesteps) * 500 + 300
        },
        'context_features': np.random.rand(n_samples, n_timesteps, context_dim),
        'targets': np.random.rand(n_samples, n_timesteps) * 400 + 200
    }
    
    registry = {
        'solar': './checkpoints/solar/best.pt',
        'wind': './checkpoints/wind/best.pt',
        'battery_soc': './checkpoints/battery/best.pt',
        'grid_voltage': './checkpoints/grid/best.pt',
        'load_forecast': './checkpoints/load/best.pt'
    }
    
    params = {'epochs': 30, 'learning_rate': 1e-3, 'batch_size': 32}
    
    try:
        model, summary = train_ensemble_meta_model(mock_data, registry, params)
        print(f"✅ Test Successful: {summary['best_val_loss']:.4f}")
        print(f"   Learned Weights: {summary['learned_weights']}")
        print(f"   Mean Uncertainty: {summary['physics_validation']['mean_uncertainty']:.4f}")
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        raise
