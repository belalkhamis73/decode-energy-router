"""
Enhanced PINN Engine - Multi-Model Neural Network Inference System
Supports: Solar, Wind, Battery Thermal, Grid Dynamics, Load Forecasting
Features: Model Versioning, A/B Testing, Physics Validation, Uncertainty Quantification
"""

import torch
import torch.nn.functional as F
import logging
import os
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Configure Logging
logger = logging.getLogger("PINN_Engine")

# --- Import Architecture (with Mock Fallback) ---
try:
    from ml_models.architectures.deeponet import DeepONet
except ImportError:
    from torch import nn
    class DeepONet(nn.Module):
        def __init__(self, n_buses=14, input_dim=3, hidden_dim=64, output_dim=1): 
            super().__init__()
            self.n_buses = n_buses
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),  # For uncertainty estimation
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x): 
            return self.fc(x)


class ModelType(Enum):
    """Enumeration of supported PINN model types"""
    SOLAR = "solar"
    WIND = "wind"
    BATTERY_THERMAL = "battery_thermal"
    BATTERY_SOH = "battery_soh"
    GRID_VOLTAGE = "grid_voltage"
    GRID_FREQUENCY = "grid_frequency"
    LOAD_FORECAST = "load_forecast"


@dataclass
class ModelMetrics:
    """Performance tracking for individual models"""
    inference_count: int = 0
    total_inference_time: float = 0.0
    avg_physics_residual: float = 0.0
    last_accuracy: float = 0.0
    uncertainty_avg: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_inference_time(self) -> float:
        return self.total_inference_time / max(self.inference_count, 1)


@dataclass
class ModelInstance:
    """Container for model with metadata"""
    model: torch.nn.Module
    version: str
    loaded_at: datetime
    metrics: ModelMetrics
    artifact_path: Optional[str] = None
    is_bayesian: bool = False  # Enable dropout at inference for uncertainty


class PhysicsValidator:
    """Validates model outputs against physics constraints"""
    
    @staticmethod
    def solar_residual(ghi: float, temp: float, output: float) -> float:
        """
        Solar PV physics: P = η * A * GHI * temp_factor
        Residual checks if output violates physical bounds
        """
        # Typical panel efficiency 15-22%, let's assume max 25% with ideal conditions
        max_power = 0.25 * ghi  # Simplified: max power per unit irradiance
        temp_derate = 1.0 - 0.004 * max(0, temp - 25)  # -0.4%/°C above 25°C
        max_realistic = max_power * temp_derate
        
        # Residual: how much does output exceed physical maximum?
        violation = max(0, output - max_realistic)
        return abs(violation) + abs(min(0, output))  # Penalize negative too
    
    @staticmethod
    def wind_residual(wind_speed: float, output: float, rated_power: float = 1.0) -> float:
        """
        Wind turbine power curve: P = 0.5 * ρ * A * Cp * v³
        Simplified: Betz limit = 59.3% max efficiency
        """
        # Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s
        if wind_speed < 3.0:
            expected_max = 0.0
        elif wind_speed > 25.0:
            expected_max = 0.0  # Turbine shutdown
        elif wind_speed > 12.0:
            expected_max = rated_power
        else:
            # Cubic relationship in operational range
            expected_max = rated_power * ((wind_speed - 3) / 9) ** 3
        
        expected_max *= 0.593  # Betz limit
        violation = max(0, output - expected_max)
        return abs(violation) + abs(min(0, output))
    
    @staticmethod
    def battery_thermal_residual(soc: float, current: float, temp: float) -> float:
        """
        Battery thermal: Q = I²R + ΔH_reaction
        Temperature should correlate with current²
        """
        # Expected heat generation (simplified)
        expected_temp_rise = 0.1 * current**2  # Internal resistance heating
        expected_temp = 25.0 + expected_temp_rise
        
        # Residual: temperature deviation from physics
        return abs(temp - expected_temp) / max(expected_temp, 1.0)
    
    @staticmethod
    def grid_voltage_residual(load: float, generation: float, voltage: float) -> float:
        """
        Grid voltage: Should stay near 1.0 p.u. (0.95 - 1.05 acceptable)
        Large imbalances should cause deviation
        """
        imbalance = abs(generation - load)
        expected_deviation = min(0.05, 0.01 * imbalance)  # Max 5% deviation
        
        voltage_dev = abs(voltage - 1.0)
        if voltage_dev > expected_deviation:
            return voltage_dev - expected_deviation
        return 0.0
    
    @staticmethod
    def grid_frequency_residual(power_imbalance: float, frequency: float) -> float:
        """
        Grid frequency: f = f0 ± Δf (where Δf ∝ ΔP/inertia)
        Should stay 59.5 - 60.5 Hz (49.5 - 50.5 for 50Hz systems)
        """
        nominal = 60.0
        max_deviation = 0.5
        expected_freq = nominal - np.clip(power_imbalance * 0.1, -max_deviation, max_deviation)
        
        return abs(frequency - expected_freq) / nominal


class PINNEngine:
    """
    Multi-Model Physics-Informed Neural Network Engine
    Manages specialized models for different energy system components
    """
    
    def __init__(self, enable_gpu: bool = True):
        self.device = torch.device("cuda" if enable_gpu and torch.cuda.is_available() else "cpu")
        
        # Multi-level registry: model_type -> session_id -> ModelInstance
        self.models: Dict[str, Dict[str, ModelInstance]] = {
            model_type.value: {} for model_type in ModelType
        }
        
        # A/B Testing: session_id -> {model_type: [version_a, version_b]}
        self.ab_tests: Dict[str, Dict[str, List[str]]] = {}
        
        # Physics validator
        self.physics = PhysicsValidator()
        
        logger.info(f"🚀 Enhanced PINN Engine Initialized on {self.device}")
        logger.info(f"📊 Tracking {len(ModelType)} model types")
    
    # ==================== MODEL REGISTRATION ====================
    
    def register_model(
        self,
        session_id: str,
        model_type: ModelType,
        input_dim: int,
        output_dim: int,
        version: str = "v1.0",
        artifact_path: Optional[str] = None,
        hidden_dim: int = 64,
        n_buses: Optional[int] = None,
        enable_bayesian: bool = False
    ) -> bool:
        """
        Register a specialized model for a session
        
        Args:
            session_id: Unique session identifier
            model_type: Type of model (solar, wind, etc.)
            input_dim: Input feature dimension
            output_dim: Output dimension
            version: Model version tag
            artifact_path: Path to pretrained weights
            hidden_dim: Hidden layer size
            n_buses: For grid models, number of buses
            enable_bayesian: Enable dropout for uncertainty estimation
        """
        try:
            logger.info(f"🧠 Registering {model_type.value} model for session {session_id} (v{version})")
            
            # Initialize architecture based on model type
            if model_type in [ModelType.GRID_VOLTAGE, ModelType.GRID_FREQUENCY]:
                if n_buses is None:
                    raise ValueError(f"{model_type.value} requires n_buses parameter")
                model = DeepONet(
                    n_buses=n_buses,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
            else:
                model = DeepONet(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
            
            # Load pretrained weights if available
            if artifact_path and os.path.exists(artifact_path):
                state_dict = torch.load(artifact_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"   ✓ Loaded weights from {artifact_path}")
            
            # Move to device and set mode
            model.to(self.device)
            if enable_bayesian:
                model.train()  # Keep dropout active
            else:
                model.eval()
            
            # Create model instance with metadata
            model_instance = ModelInstance(
                model=model,
                version=version,
                loaded_at=datetime.now(),
                metrics=ModelMetrics(),
                artifact_path=artifact_path,
                is_bayesian=enable_bayesian
            )
            
            # Store in registry
            self.models[model_type.value][session_id] = model_instance
            logger.info(f"   ✓ Model registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register {model_type.value} model: {e}")
            return False
    
    def setup_ab_test(
        self,
        session_id: str,
        model_type: ModelType,
        version_a: str,
        version_b: str
    ):
        """Configure A/B test between two model versions"""
        if session_id not in self.ab_tests:
            self.ab_tests[session_id] = {}
        
        self.ab_tests[session_id][model_type.value] = [version_a, version_b]
        logger.info(f"🔬 A/B test configured: {model_type.value} ({version_a} vs {version_b})")
    
    # ==================== SPECIALIZED INFERENCE METHODS ====================
    
    def infer_solar_production(
        self,
        session_id: str,
        ghi: float,
        temp: float,
        panel_angle: float,
        cloud_override: Optional[float] = None,
        return_uncertainty: bool = False
    ) -> Tuple[float, Optional[float], float]:
        """
        Predict solar PV production
        
        Args:
            ghi: Global Horizontal Irradiance (W/m²)
            temp: Panel temperature (°C)
            panel_angle: Panel tilt angle (degrees)
            cloud_override: Manual cloud cover factor (0-1)
            return_uncertainty: Compute prediction uncertainty
        
        Returns:
            (power_output, uncertainty, physics_residual)
        """
        # Prepare input tensor
        cloud_factor = cloud_override if cloud_override is not None else 0.0
        input_tensor = torch.tensor([[ghi, temp, panel_angle, cloud_factor]], dtype=torch.float32)
        
        # Get prediction
        output, uncertainty = self._infer(
            session_id, ModelType.SOLAR, input_tensor, return_uncertainty
        )
        power = output.item()
        
        # Validate physics
        residual = self.physics.solar_residual(ghi, temp, power)
        
        return power, uncertainty, residual
    
    def infer_wind_production(
        self,
        session_id: str,
        wind_speed: float,
        air_density: float = 1.225,
        turbine_state: int = 1,
        return_uncertainty: bool = False
    ) -> Tuple[float, Optional[float], float]:
        """
        Predict wind turbine production
        
        Args:
            wind_speed: Wind speed at hub height (m/s)
            air_density: Air density (kg/m³)
            turbine_state: Turbine operational state (1=on, 0=off)
            return_uncertainty: Compute prediction uncertainty
        
        Returns:
            (power_output, uncertainty, physics_residual)
        """
        input_tensor = torch.tensor(
            [[wind_speed, air_density, float(turbine_state)]],
            dtype=torch.float32
        )
        
        output, uncertainty = self._infer(
            session_id, ModelType.WIND, input_tensor, return_uncertainty
        )
        power = output.item()
        
        # Validate against wind power curve
        residual = self.physics.wind_residual(wind_speed, power)
        
        return power, uncertainty, residual
    
    def infer_battery_thermal(
        self,
        session_id: str,
        soc: float,
        current: float,
        temp_ambient: float,
        cooling_factor: float = 1.0,
        return_uncertainty: bool = False
    ) -> Tuple[float, Optional[float], float]:
        """
        Predict battery cell temperature
        
        Args:
            soc: State of charge (0-1)
            current: Charge/discharge current (A)
            temp_ambient: Ambient temperature (°C)
            cooling_factor: Cooling system effectiveness (0-1)
            return_uncertainty: Compute prediction uncertainty
        
        Returns:
            (cell_temperature, uncertainty, physics_residual)
        """
        input_tensor = torch.tensor(
            [[soc, current, temp_ambient, cooling_factor]],
            dtype=torch.float32
        )
        
        output, uncertainty = self._infer(
            session_id, ModelType.BATTERY_THERMAL, input_tensor, return_uncertainty
        )
        temp = output.item()
        
        # Validate thermal physics
        residual = self.physics.battery_thermal_residual(soc, current, temp)
        
        return temp, uncertainty, residual
    
    def infer_grid_voltage(
        self,
        session_id: str,
        load: torch.Tensor,
        generation: torch.Tensor,
        topology: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Predict grid bus voltages
        
        Args:
            load: Load vector [n_buses] (p.u.)
            generation: Generation vector [n_buses] (p.u.)
            topology: Optional network topology features
            return_uncertainty: Compute prediction uncertainty
        
        Returns:
            (voltage_vector, uncertainty, physics_residual)
        """
        # Combine inputs
        if topology is not None:
            input_tensor = torch.cat([load, generation, topology], dim=-1)
        else:
            input_tensor = torch.cat([load, generation], dim=-1)
        
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        output, uncertainty = self._infer(
            session_id, ModelType.GRID_VOLTAGE, input_tensor, return_uncertainty
        )
        
        # Validate voltage constraints
        total_load = load.sum().item()
        total_gen = generation.sum().item()
        avg_voltage = output.mean().item()
        residual = self.physics.grid_voltage_residual(total_load, total_gen, avg_voltage)
        
        return output, uncertainty, residual
    
    def infer_grid_frequency(
        self,
        session_id: str,
        power_imbalance: float,
        inertia: float,
        return_uncertainty: bool = False
    ) -> Tuple[float, Optional[float], float]:
        """
        Predict grid frequency
        
        Args:
            power_imbalance: Generation - Load (p.u.)
            inertia: System inertia constant (s)
            return_uncertainty: Compute prediction uncertainty
        
        Returns:
            (frequency_hz, uncertainty, physics_residual)
        """
        input_tensor = torch.tensor([[power_imbalance, inertia]], dtype=torch.float32)
        
        output, uncertainty = self._infer(
            session_id, ModelType.GRID_FREQUENCY, input_tensor, return_uncertainty
        )
        frequency = output.item()
        
        # Validate frequency dynamics
        residual = self.physics.grid_frequency_residual(power_imbalance, frequency)
        
        return frequency, uncertainty, residual
    
    def infer_load_forecast(
        self,
        session_id: str,
        historical_window: torch.Tensor,
        weather: torch.Tensor,
        time_features: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Forecast future load
        
        Args:
            historical_window: Past load values [window_size]
            weather: Weather forecast features [n_features]
            time_features: Time encoding (hour, day, month) [3]
            return_uncertainty: Compute prediction uncertainty
        
        Returns:
            (load_forecast, uncertainty, physics_residual)
        """
        # Combine all features
        input_tensor = torch.cat([
            historical_window.flatten(),
            weather.flatten(),
            time_features.flatten()
        ]).unsqueeze(0)
        
        output, uncertainty = self._infer(
            session_id, ModelType.LOAD_FORECAST, input_tensor, return_uncertainty
        )
        
        # Basic residual: check for unrealistic load values
        residual = torch.relu(output - 10.0).sum().item()  # Penalty if > 10 p.u.
        residual += torch.relu(-output).sum().item()  # Penalty if negative
        
        return output, uncertainty, residual
    
    # ==================== CORE INFERENCE ENGINE ====================
    
    def _infer(
        self,
        session_id: str,
        model_type: ModelType,
        input_tensor: torch.Tensor,
        return_uncertainty: bool = False,
        n_samples: int = 20
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Core inference method with uncertainty quantification
        
        Args:
            session_id: Session identifier
            model_type: Model type to use
            input_tensor: Input features
            return_uncertainty: Compute epistemic uncertainty
            n_samples: Number of Monte Carlo samples for uncertainty
        
        Returns:
            (prediction, uncertainty_std)
        """
        model_key = model_type.value
        
        # Check if model exists
        if session_id not in self.models[model_key]:
            logger.warning(f"⚠️  No {model_key} model for session {session_id}, using fallback")
            return self._fallback_prediction(model_type, input_tensor), None
        
        # Get model instance
        model_inst = self.models[model_key][session_id]
        model = model_inst.model
        
        # Timing
        start_time = time.time()
        
        try:
            input_tensor = input_tensor.to(self.device)
            
            # Standard inference
            if not return_uncertainty or not model_inst.is_bayesian:
                with torch.no_grad():
                    output = model(input_tensor).cpu()
                uncertainty = None
            
            # Bayesian inference (MC Dropout)
            else:
                predictions = []
                for _ in range(n_samples):
                    with torch.no_grad():
                        pred = model(input_tensor).cpu()
                        predictions.append(pred)
                
                predictions = torch.stack(predictions)
                output = predictions.mean(dim=0)
                uncertainty = predictions.std(dim=0)
            
            # Update metrics
            inference_time = time.time() - start_time
            model_inst.metrics.inference_count += 1
            model_inst.metrics.total_inference_time += inference_time
            
            if uncertainty is not None:
                model_inst.metrics.uncertainty_avg = uncertainty.mean().item()
            
            return output, uncertainty
            
        except Exception as e:
            logger.error(f"❌ Inference error for {model_key}: {e}")
            return self._fallback_prediction(model_type, input_tensor), None
    
    def _fallback_prediction(self, model_type: ModelType, input_tensor: torch.Tensor) -> torch.Tensor:
        """Generate physics-based fallback when model unavailable"""
        if model_type == ModelType.SOLAR:
            return torch.ones(input_tensor.shape[0], 1) * 0.2  # 20% efficiency baseline
        elif model_type == ModelType.WIND:
            return torch.ones(input_tensor.shape[0], 1) * 0.3  # 30% capacity factor
        elif model_type == ModelType.BATTERY_THERMAL:
            return torch.ones(input_tensor.shape[0], 1) * 25.0  # 25°C nominal
        elif model_type == ModelType.GRID_VOLTAGE:
            return torch.ones(input_tensor.shape[0], 1) * 1.0  # 1.0 p.u. nominal
        elif model_type == ModelType.GRID_FREQUENCY:
            return torch.ones(input_tensor.shape[0], 1) * 60.0  # 60 Hz nominal
        else:
            return torch.ones(input_tensor.shape[0], 1) * 0.5  # Generic fallback
    
    # ==================== GRADIENT & SENSITIVITY ANALYSIS ====================
    
    def compute_sensitivity(
        self,
        session_id: str,
        model_type: ModelType,
        input_tensor: torch.Tensor,
        output_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute gradient of output w.r.t. inputs (sensitivity analysis)
        
        Returns:
            Gradient tensor [input_dim]
        """
        model_key = model_type.value
        if session_id not in self.models[model_key]:
            return torch.zeros_like(input_tensor)
        
        model = self.models[model_key][session_id].model
        model.eval()
        
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        
        output = model(input_tensor)
        output[0, output_idx].backward()
        
        return input_tensor.grad.cpu()
    
    # ==================== METRICS & DIAGNOSTICS ====================
    
    def get_model_metrics(self, session_id: str, model_type: ModelType) -> Optional[ModelMetrics]:
        """Retrieve performance metrics for a model"""
        model_key = model_type.value
        if session_id in self.models[model_key]:
            return self.models[model_key][session_id].metrics
        return None
    
    def get_all_metrics(self, session_id: str) -> Dict[str, ModelMetrics]:
        """Get metrics for all models in a session"""
        metrics = {}
        for model_type in ModelType:
            m = self.get_model_metrics(session_id, model_type)
            if m:
                metrics[model_type.value] = m
        return metrics
    
    def log_diagnostics(self, session_id: str):
        """Log diagnostic information for all models"""
        logger.info(f"\n📊 Diagnostics for Session: {session_id}")
        for model_type in ModelType:
            metrics = self.get_model_metrics(session_id, model_type)
            if metrics:
                logger.info(f"  {model_type.value}:")
                logger.info(f"    Inferences: {metrics.inference_count}")
                logger.info(f"    Avg Time: {metrics.avg_inference_time*1000:.2f} ms")
                logger.info(f"    Uncertainty: {metrics.uncertainty_avg:.4f}")


# ==================== SINGLETON INSTANCE ====================
pinn_engine = PINNEngine()
