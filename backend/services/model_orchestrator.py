"""
backend/services/model_orchestrator.py

Central coordinator for multi-model ensemble execution with health monitoring,
hot-swapping capabilities, and explainability features.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict
import json
import time

# Optional imports with graceful fallbacks
try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    import psycopg2
    from psycopg2.extras import Json
except ImportError:
    psycopg2 = None

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of available model types."""
    PINN_BASE = "pinn_base"
    PINN_ENHANCED = "pinn_enhanced"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_ODE = "neural_ode"


class ModelStatus(Enum):
    """Model health status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    SWAPPING = "swapping"


@dataclass
class ModelMetrics:
    """Metrics for individual model performance."""
    model_type: ModelType
    inference_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    last_inference_time: Optional[datetime] = None
    status: ModelStatus = ModelStatus.LOADING
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average inference latency."""
        if self.inference_count == 0:
            return 0.0
        return self.total_latency_ms / self.inference_count
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.inference_count == 0:
            return 0.0
        return self.error_count / self.inference_count


@dataclass
class PredictionResult:
    """Container for model prediction results."""
    model_type: ModelType
    prediction: np.ndarray
    confidence: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Aggregated prediction from multiple models."""
    mean_prediction: np.ndarray
    median_prediction: np.ndarray
    std_prediction: np.ndarray
    confidence_interval: Tuple[np.ndarray, np.ndarray]
    individual_predictions: List[PredictionResult]
    weights: Dict[ModelType, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ModelRegistry:
    """
    Tracks available models per session with health monitoring.
    """
    
    def __init__(self):
        self.session_models: Dict[str, Dict[ModelType, Any]] = {}
        self.model_metrics: Dict[str, Dict[ModelType, ModelMetrics]] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def register_session(
        self, 
        session_id: str, 
        grid_config: Dict[str, Any],
        model_instances: Dict[ModelType, Any]
    ) -> bool:
        """
        Register models for a specific session.
        
        Args:
            session_id: Unique session identifier
            grid_config: Grid configuration parameters
            model_instances: Dictionary of initialized model instances
            
        Returns:
            True if registration successful
        """
        async with self._lock:
            try:
                self.session_models[session_id] = model_instances
                self.model_configs[session_id] = grid_config
                
                # Initialize metrics for each model
                self.model_metrics[session_id] = {
                    model_type: ModelMetrics(model_type=model_type, status=ModelStatus.HEALTHY)
                    for model_type in model_instances.keys()
                }
                
                logger.info(
                    f"Registered {len(model_instances)} models for session {session_id}"
                )
                return True
                
            except Exception as e:
                logger.error(f"Failed to register session {session_id}: {e}")
                return False
    
    async def get_session_models(
        self, 
        session_id: str
    ) -> Optional[Dict[ModelType, Any]]:
        """Retrieve models for a session."""
        return self.session_models.get(session_id)
    
    async def update_model_metrics(
        self,
        session_id: str,
        model_type: ModelType,
        latency_ms: float,
        success: bool = True
    ):
        """Update performance metrics for a model."""
        if session_id not in self.model_metrics:
            return
        
        metrics = self.model_metrics[session_id][model_type]
        metrics.inference_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.last_inference_time = datetime.utcnow()
        
        if not success:
            metrics.error_count += 1
            
            # Degrade status if error rate too high
            if metrics.error_rate > 0.1:  # 10% threshold
                metrics.status = ModelStatus.DEGRADED
        
        # Mark as healthy if error rate improves
        if metrics.error_rate < 0.05 and metrics.status == ModelStatus.DEGRADED:
            metrics.status = ModelStatus.HEALTHY
    
    async def get_metrics(
        self, 
        session_id: str
    ) -> Dict[ModelType, ModelMetrics]:
        """Get metrics for all models in a session."""
        return self.model_metrics.get(session_id, {})
    
    async def remove_session(self, session_id: str):
        """Clean up session resources."""
        async with self._lock:
            self.session_models.pop(session_id, None)
            self.model_metrics.pop(session_id, None)
            self.model_configs.pop(session_id, None)
            logger.info(f"Removed session {session_id}")


class PredictionAggregator:
    """
    Combines multi-model outputs using various aggregation strategies.
    """
    
    def __init__(self, aggregation_method: str = "weighted_average"):
        """
        Initialize aggregator.
        
        Args:
            aggregation_method: One of 'weighted_average', 'median', 'stacking'
        """
        self.method = aggregation_method
    
    def aggregate(
        self,
        predictions: List[PredictionResult],
        weights: Optional[Dict[ModelType, float]] = None
    ) -> EnsemblePrediction:
        """
        Aggregate predictions from multiple models.
        
        Args:
            predictions: List of individual model predictions
            weights: Optional weights for each model type
            
        Returns:
            EnsemblePrediction with aggregated results
        """
        if not predictions:
            raise ValueError("No predictions to aggregate")
        
        # Extract prediction arrays
        pred_arrays = [p.prediction for p in predictions]
        
        # Default to equal weights
        if weights is None:
            weights = {p.model_type: 1.0 / len(predictions) for p in predictions}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted average
        weighted_preds = []
        for pred in predictions:
            weight = normalized_weights.get(pred.model_type, 0.0)
            weighted_preds.append(pred.prediction * weight)
        
        mean_pred = np.sum(weighted_preds, axis=0)
        
        # Calculate statistics
        pred_stack = np.stack(pred_arrays)
        median_pred = np.median(pred_stack, axis=0)
        std_pred = np.std(pred_stack, axis=0)
        
        # 95% confidence interval
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred
        
        return EnsemblePrediction(
            mean_prediction=mean_pred,
            median_prediction=median_pred,
            std_prediction=std_pred,
            confidence_interval=(ci_lower, ci_upper),
            individual_predictions=predictions,
            weights=normalized_weights
        )


class UncertaintyQuantifier:
    """
    Calculates ensemble confidence intervals and uncertainty metrics.
    """
    
    @staticmethod
    def calculate_epistemic_uncertainty(
        predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate epistemic (model) uncertainty.
        
        Args:
            predictions: List of prediction arrays from different models
            
        Returns:
            Array of epistemic uncertainty values
        """
        pred_stack = np.stack(predictions)
        return np.var(pred_stack, axis=0)
    
    @staticmethod
    def calculate_aleatoric_uncertainty(
        predictions: List[PredictionResult]
    ) -> np.ndarray:
        """
        Calculate aleatoric (data) uncertainty from individual model confidences.
        
        Args:
            predictions: List of PredictionResult objects
            
        Returns:
            Array of aleatoric uncertainty estimates
        """
        # Use inverse of confidence as uncertainty proxy
        uncertainties = [1.0 - p.confidence for p in predictions]
        return np.array(uncertainties).mean()
    
    @staticmethod
    def quantify_total_uncertainty(
        ensemble_prediction: EnsemblePrediction
    ) -> Dict[str, float]:
        """
        Compute total uncertainty metrics.
        
        Args:
            ensemble_prediction: Aggregated ensemble prediction
            
        Returns:
            Dictionary of uncertainty metrics
        """
        predictions = [p.prediction for p in ensemble_prediction.individual_predictions]
        
        epistemic = UncertaintyQuantifier.calculate_epistemic_uncertainty(predictions)
        aleatoric = UncertaintyQuantifier.calculate_aleatoric_uncertainty(
            ensemble_prediction.individual_predictions
        )
        
        return {
            'epistemic_uncertainty': float(np.mean(epistemic)),
            'aleatoric_uncertainty': float(aleatoric),
            'total_uncertainty': float(np.mean(epistemic) + aleatoric),
            'confidence_interval_width': float(
                np.mean(ensemble_prediction.confidence_interval[1] - 
                       ensemble_prediction.confidence_interval[0])
            )
        }


class ModelOrchestrator:
    """
    Main orchestration engine for multi-model ensemble execution.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        timescale_config: Optional[Dict[str, Any]] = None,
        aggregation_method: str = "weighted_average"
    ):
        """
        Initialize the model orchestrator.
        
        Args:
            redis_url: Redis connection URL for prediction streaming
            timescale_config: TimescaleDB configuration for logging
            aggregation_method: Method for combining predictions
        """
        self.registry = ModelRegistry()
        self.aggregator = PredictionAggregator(aggregation_method)
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        # Redis client for streaming predictions
        self.redis_client = None
        if redis and redis_url:
            self.redis_url = redis_url
        
        # TimescaleDB connection config
        self.timescale_config = timescale_config
        self.db_conn = None
        
        # Model weights (can be learned over time)
        self.model_weights: Dict[str, Dict[ModelType, float]] = defaultdict(
            lambda: {mt: 1.0 for mt in ModelType}
        )
    
    async def initialize(self):
        """Initialize external connections."""
        # Initialize Redis
        if redis and self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
        
        # Initialize TimescaleDB
        if psycopg2 and self.timescale_config:
            try:
                self.db_conn = psycopg2.connect(**self.timescale_config)
                logger.info("TimescaleDB connection established")
            except Exception as e:
                logger.warning(f"TimescaleDB connection failed: {e}")
    
    async def register_session_models(
        self,
        session_id: str,
        grid_config: Dict[str, Any],
        model_factory: Optional[callable] = None
    ) -> bool:
        """
        Initialize all 6 models for a session.
        
        Args:
            session_id: Unique session identifier
            grid_config: Grid configuration parameters
            model_factory: Optional factory function to create model instances
            
        Returns:
            True if all models registered successfully
        """
        try:
            # Create model instances (placeholder - integrate with actual models)
            model_instances = {}
            
            for model_type in ModelType:
                if model_factory:
                    model_instances[model_type] = await model_factory(
                        model_type, grid_config
                    )
                else:
                    # Placeholder - replace with actual model initialization
                    model_instances[model_type] = self._create_dummy_model(
                        model_type, grid_config
                    )
            
            # Register with registry
            success = await self.registry.register_session(
                session_id, grid_config, model_instances
            )
            
            if success:
                logger.info(f"All models registered for session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to register models for session {session_id}: {e}")
            return False
    
    def _create_dummy_model(self, model_type: ModelType, grid_config: Dict[str, Any]):
        """Placeholder model creation - replace with actual implementation."""
        return {
            'type': model_type,
            'config': grid_config,
            'initialized': True
        }
    
    async def run_parallel_inference(
        self,
        session_id: str,
        inputs: np.ndarray,
        user_overrides: Optional[Dict[str, Any]] = None
    ) -> EnsemblePrediction:
        """
        Execute all models concurrently using asyncio.
        
        Args:
            session_id: Session identifier
            inputs: Input data for inference
            user_overrides: Optional parameter overrides
            
        Returns:
            Aggregated ensemble prediction
        """
        models = await self.registry.get_session_models(session_id)
        
        if not models:
            raise ValueError(f"No models found for session {session_id}")
        
        # Run inference tasks in parallel
        tasks = [
            self._run_single_inference(
                session_id, model_type, model, inputs, user_overrides
            )
            for model_type, model in models.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed predictions
        successful_predictions = [
            r for r in results if isinstance(r, PredictionResult)
        ]
        
        if not successful_predictions:
            raise RuntimeError("All model inferences failed")
        
        # Log failures
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            logger.warning(f"Session {session_id}: {len(failures)} model(s) failed")
        
        # Aggregate predictions
        weights = self.model_weights.get(session_id, {})
        ensemble = self.aggregator.aggregate(successful_predictions, weights)
        
        # Publish to Redis stream
        await self._publish_prediction(session_id, ensemble)
        
        # Log to TimescaleDB
        await self._log_prediction(session_id, ensemble, inputs)
        
        return ensemble
    
    async def _run_single_inference(
        self,
        session_id: str,
        model_type: ModelType,
        model: Any,
        inputs: np.ndarray,
        overrides: Optional[Dict[str, Any]]
    ) -> PredictionResult:
        """Run inference on a single model with timing."""
        start_time = time.perf_counter()
        
        try:
            # Placeholder - replace with actual model inference
            prediction = await self._inference_placeholder(model, inputs, overrides)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            await self.registry.update_model_metrics(
                session_id, model_type, latency_ms, success=True
            )
            
            return PredictionResult(
                model_type=model_type,
                prediction=prediction,
                confidence=0.95,  # Placeholder
                latency_ms=latency_ms,
                metadata={'overrides': overrides}
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            await self.registry.update_model_metrics(
                session_id, model_type, latency_ms, success=False
            )
            logger.error(f"Inference failed for {model_type}: {e}")
            raise
    
    async def _inference_placeholder(
        self, model: Any, inputs: np.ndarray, overrides: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Placeholder for actual model inference - integrate with pinn_engine."""
        await asyncio.sleep(0.01)  # Simulate inference time
        return np.random.randn(*inputs.shape)  # Dummy prediction
    
    async def get_model_health(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return model status, load, and inference latency.
        
        Args:
            session_id: Optional session to filter by
            
        Returns:
            Dictionary of health metrics
        """
        if session_id:
            metrics = await self.registry.get_metrics(session_id)
            return {
                'session_id': session_id,
                'models': {
                    model_type.value: {
                        'status': metric.status.value,
                        'avg_latency_ms': metric.avg_latency_ms,
                        'error_rate': metric.error_rate,
                        'inference_count': metric.inference_count,
                        'last_inference': metric.last_inference_time.isoformat()
                        if metric.last_inference_time else None
                    }
                    for model_type, metric in metrics.items()
                }
            }
        else:
            # Return health for all sessions
            all_health = {}
            for sid in self.registry.session_models.keys():
                all_health[sid] = await self.get_model_health(sid)
            return all_health
    
    async def hot_swap_model(
        self,
        session_id: str,
        model_type: ModelType,
        new_checkpoint: str
    ) -> bool:
        """
        Live model replacement without downtime.
        
        Args:
            session_id: Session identifier
            model_type: Type of model to swap
  
