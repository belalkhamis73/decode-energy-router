"""
Ensemble Meta-Learner for Digital Twin.
Learns optimal combination weights for multiple physics-informed models.
Implements dynamic weighting based on context and model confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class ContextAwareWeightingNetwork(nn.Module):
    """
    Neural network that learns to weight different models based on input context.
    Key insight: Different models excel in different operating conditions.
    
    Example: Solar model is most reliable during clear days, while ensemble
    should rely more on historical load patterns during cloudy weather.
    """
    def __init__(self, context_dim: int, num_models: int):
        super().__init__()
        self.num_models = num_models
        
        # Attention-like mechanism for model selection
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [Batch, Context_Dim] - Current operating conditions
        Returns:
            weights: [Batch, Num_Models] - Normalized weights (sum to 1)
        """
        logits = self.context_encoder(context)
        weights = F.softmax(logits, dim=-1)
        return weights

class UncertaintyAwareWeighting(nn.Module):
    """
    Weights models inversely proportional to their predicted uncertainty.
    Models that are more confident get higher weight.
    
    Based on inverse variance weighting: w_i = 1/σ_i² / Σ(1/σ_j²)
    """
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uncertainties: [Batch, Num_Models] - Standard deviations from each model
        Returns:
            weights: [Batch, Num_Models] - Inverse variance weights
        """
        # Prevent division by zero
        variances = uncertainties ** 2 + self.epsilon
        
        # Inverse variance weights
        inv_var = 1.0 / variances
        weights = inv_var / inv_var.sum(dim=-1, keepdim=True)
        
        return weights

class PerformanceTracking(nn.Module):
    """
    Tracks historical performance of each model using exponential moving average.
    Models with better recent performance get higher base weights.
    """
    def __init__(self, num_models: int, decay: float = 0.95):
        super().__init__()
        self.decay = decay
        
        # Running averages of performance (lower is better)
        self.register_buffer("performance_ema", torch.ones(num_models))
        self.register_buffer("n_updates", torch.zeros(1))
    
    def update(self, errors: torch.Tensor):
        """
        Args:
            errors: [Num_Models] - Recent error for each model
        """
        if self.training:
            self.performance_ema = (
                self.decay * self.performance_ema + 
                (1 - self.decay) * errors
            )
            self.n_updates += 1
    
    def get_weights(self) -> torch.Tensor:
        """
        Returns performance-based weights.
        Better performers (lower error) get higher weight.
        """
        # Inverse of performance (lower error = higher weight)
        inv_perf = 1.0 / (self.performance_ema + 1e-6)
        weights = inv_perf / inv_perf.sum()
        return weights

class EnsembleMetaLearner(nn.Module):
    """
    Master ensemble that combines predictions from multiple specialized models:
    - Solar PINN
    - Wind PINN
    - Battery Thermal PINN
    - Frequency PINN
    - Load Forecasting Model
    
    Combines three weighting strategies:
    1. Context-aware (neural network learns from data)
    2. Uncertainty-aware (trust confident models more)
    3. Performance-aware (trust historically accurate models)
    """
    def __init__(self, context_dim: int, num_models: int = 5, 
                 combination_mode: str = "learned"):
        super().__init__()
        self.num_models = num_models
        self.combination_mode = combination_mode
        
        # Weighting strategies
        self.context_weights = ContextAwareWeightingNetwork(context_dim, num_models)
        self.uncertainty_weights = UncertaintyAwareWeighting()
        self.performance_tracker = PerformanceTracking(num_models)
        
        # Meta-parameters for combining different weighting strategies
        self.register_parameter(
            "strategy_weights",
            nn.Parameter(torch.ones(3) / 3.0)  # [context, uncertainty, performance]
        )
        
    def forward(self, predictions: torch.Tensor, uncertainties: torch.Tensor,
               context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: [Batch, Num_Models, Output_Dim] - Predictions from each model
            uncertainties: [Batch, Num_Models, Output_Dim] - Uncertainties from each model
            context: [Batch, Context_Dim] - Operating conditions
            
        Returns:
            ensemble_pred: [Batch, Output_Dim] - Weighted prediction
            ensemble_uncertainty: [Batch, Output_Dim] - Combined uncertainty
            weight_info: Dictionary with weight decomposition for interpretability
        """
        batch_size = predictions.shape[0]
        output_dim = predictions.shape[2]
        
        # Strategy 1: Context-aware weights
        w_context = self.context_weights(context)  # [B, M]
        
        # Strategy 2: Uncertainty-aware weights (averaged across output dims)
        uncertainties_avg = uncertainties.mean(dim=-1)  # [B, M]
        w_uncertainty = self.uncertainty_weights(uncertainties_avg)
        
        # Strategy 3: Performance-based weights
        w_performance = self.performance_tracker.get_weights()  # [M]
        w_performance = w_performance.unsqueeze(0).expand(batch_size, -1)
        
        # Normalize strategy weights
        strategy_weights = F.softmax(self.strategy_weights, dim=0)
        
        # Combine strategies
        if self.combination_mode == "learned":
            final_weights = (
                strategy_weights[0] * w_context +
                strategy_weights[1] * w_uncertainty +
                strategy_weights[2] * w_performance
            )
        elif self.combination_mode == "uncertainty_only":
            final_weights = w_uncertainty
        elif self.combination_mode == "context_only":
            final_weights = w_context
        else:  # equal
            final_weights = torch.ones_like(w_context) / self.num_models
        
        # Normalize final weights
        final_weights = final_weights / final_weights.sum(dim=-1, keepdim=True)
        
        # Apply weights to predictions
        weighted_preds = predictions * final_weights.unsqueeze(-1)  # [B, M, D]
        ensemble_pred = weighted_preds.sum(dim=1)  # [B, D]
        
        # Compute ensemble uncertainty (law of total variance)
        # Var(Y) = E[Var(Y|Model)] + Var(E[Y|Model])
        
        # Within-model uncertainty (aleatoric)
        weighted_vars = (uncertainties ** 2) * final_weights.unsqueeze(-1)
        aleatoric_var = weighted_vars.sum(dim=1)
        
        # Between-model uncertainty (epistemic)
        pred_deviations = predictions - ensemble_pred.unsqueeze(1)
        epistemic_var = (pred_deviations ** 2 * final_weights.unsqueeze(-1)).sum(dim=1)
        
        # Total uncertainty
        total_var = aleatoric_var + epistemic_var
        ensemble_uncertainty = torch.sqrt(total_var + 1e-6)
        
        # Store weight information for interpretability
        weight_info = {
            'final_weights': final_weights,
            'context_weights': w_context,
            'uncertainty_weights': w_uncertainty,
            'performance_weights': w_performance,
            'strategy_weights': strategy_weights,
            'aleatoric_uncertainty': torch.sqrt(aleatoric_var),
            'epistemic_uncertainty': torch.sqrt(epistemic_var)
        }
        
        return ensemble_pred, ensemble_uncertainty, weight_info
    
    def physics_consistency_loss(self, predictions: torch.Tensor, 
                                weight_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Regularization: Predictions from physics-informed models should be similar
        when physics constraints are tight (e.g., all models should predict ~0 power
        at night for solar).
        """
        # Compute variance across model predictions
        pred_variance = predictions.var(dim=1).mean()
        
        # Penalize high variance when we're confident (low ensemble uncertainty)
        # This encourages physics consensus
        return pred_variance
    
    def explain_ensemble_decision(self, weight_info: Dict[str, torch.Tensor], 
                                 model_names: List[str]) -> Dict[str, any]:
        """
        Provides human-readable explanation of ensemble decision.
        
        Args:
            weight_info: Output from forward pass
            model_names: Names of the models in the ensemble
            
        Returns:
            Dictionary with interpretation
        """
        final_weights = weight_info['final_weights']
        
        # Average weights across batch
        avg_weights = final_weights.mean(dim=0).detach().cpu().numpy()
        
        # Rank models by importance
        ranked_indices = np.argsort(avg_weights)[::-1]
        
        explanation = {
            'model_rankings': [
                {
                    'model': model_names[i],
                    'weight': float(avg_weights[i]),
                    'contribution': f"{avg_weights[i]*100:.1f}%"
                }
                for i in ranked_indices
            ],
            'primary_model': model_names[ranked_indices[0]],
            'strategy_breakdown': {
                'context_influence': float(weight_info['strategy_weights'][0].item()),
                'uncertainty_influence': float(weight_info['strategy_weights'][1].item()),
                'performance_influence': float(weight_info['strategy_weights'][2].item())
            },
            'uncertainty_breakdown': {
                'aleatoric': float(weight_info['aleatoric_uncertainty'].mean().item()),
                'epistemic': float(weight_info['epistemic_uncertainty'].mean().item())
            }
        }
        
        return explanation
    
    def compute_diversity_score(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Measures prediction diversity across models.
        High diversity is good for ensemble robustness but should be balanced
        with physics consistency.
        """
        # Pairwise distances between model predictions
        pred_flat = predictions.flatten(start_dim=1)  # [B, M*D]
        distances = torch.cdist(pred_flat, pred_flat, p=2)  # [B, B]
        
        # Average pairwise distance (excluding diagonal)
        mask = ~torch.eye(distances.shape[1], dtype=torch.bool, device=distances.device)
        diversity = distances[:, mask].mean()
        
        return diversity

# --- Unit Test ---
if __name__ == "__main__":
    torch.manual_seed(42)
    
    batch_size = 8
    num_models = 5
    output_dim = 10  # e.g., predicting 10 future timesteps
    context_dim = 15  # e.g., weather + time features
    
    # Simulate predictions from different models
    predictions = torch.randn(batch_size, num_models, output_dim) * 10 + 50
    
    # Simulate uncertainties (some models more confident than others)
    uncertainties = torch.rand(batch_size, num_models, output_dim) * 5 + 1
    
    # Context (operating conditions)
    context = torch.randn(batch_size, context_dim)
    
    # Model names
    model_names = ['Solar PINN', 'Wind PINN', 'Battery PINN', 'Frequency PINN', 'Load Forecast']
    
    # Create ensemble
    ensemble = EnsembleMetaLearner(
        context_dim=context_dim,
        num_models=num_models,
        combination_mode="learned"
    )
    
    # Forward pass
    ensemble_pred, ensemble_unc, weight_info = ensemble(predictions, uncertainties, context)
    
    print("=" * 60)
    print("ENSEMBLE META-LEARNER TEST")
    print("=" * 60)
    
    print(f"\nInput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Uncertainties: {uncertainties.shape}")
    print(f"  Context: {context.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  Ensemble prediction: {ensemble_pred.shape}")
    print(f"  Ensemble uncertainty: {ensemble_unc.shape}")
    
    print(f"\nSample predictions (first batch):")
    for i, name in enumerate(model_names):
        print(f"  {name}: {predictions[0, i, 0].item():.2f} ± {uncertainties[0, i, 0].item():.2f}")
    print(f"  Ensemble: {ensemble_pred[0, 0].item():.2f} ± {ensemble_unc[0, 0].item():.2f}")
    
    # Model weights
    print(f"\nModel weights (first batch):")
    for i, name in enumerate(model_names):
        weight = weight_info['final_weights'][0, i].item()
        print(f"  {name}: {weight:.3f} ({weight*100:.1f}%)")
    
    # Strategy weights
    print(f"\nWeighting strategy influence:")
    strategies = ['Context-aware', 'Uncertainty-aware', 'Performance-based']
    for i, strategy in enumerate(strategies):
        weight = weight_info['strategy_weights'][i].item()
        print(f"  {strategy}: {weight:.3f} ({weight*100:.1f}%)")
    
    # Uncertainty decomposition
    print(f"\nUncertainty decomposition (first batch):")
    print(f"  Aleatoric (model confidence): {weight_info['aleatoric_uncertainty'][0, 0].item():.2f}")
    print(f"  Epistemic (model disagreement): {weight_info['epistemic_uncertainty'][0, 0].item():.2f}")
    print(f"  Total: {ensemble_unc[0, 0].item():.2f}")
    
    # Explainability
    explanation = ensemble.explain_ensemble_decision(weight_info, model_names)
    
    print(f"\nEnsemble Decision Explanation:")
    print(f"  Primary model: {explanation['primary_model']}")
    print(f"  Model rankings:")
    for rank in explanation['model_rankings'][:3]:
        print(f"    {rank['model']}: {rank['contribution']}")
    
    # Diversity score
    diversity = ensemble.compute_diversity_score(predictions)
    print(f"\nPrediction diversity score: {diversity.item():.2f}")
    
    # Physics consistency loss
    physics_loss = ensemble.physics_consistency_loss(predictions, weight_info)
    print(f"Physics consistency loss: {physics_loss.item():.4f}")
    
    # Test performance tracking
    print(f"\n" + "=" * 60)
    print("Testing Performance Tracking")
    print("=" * 60)
    
    # Simulate some model errors
    fake_errors = torch.tensor([0.5, 0.8, 0.3, 1.2, 0.6])  # Model 3 is best
    
    ensemble.train()
    for _ in range(10):
        ensemble.performance_tracker.update(fake_errors)
    ensemble.eval()
    
    perf_weights = ensemble.performance_tracker.get_weights()
    print(f"\nPerformance-based weights after tracking:")
    for i, name in enumerate(model_names):
        print(f"  {name}: {perf_weights[i].item():.3f} ({perf_weights[i].item()*100:.1f}%)")
    
    best_model_idx = perf_weights.argmax().item()
    print(f"\nBest performing model: {model_names[best_model_idx]}")
