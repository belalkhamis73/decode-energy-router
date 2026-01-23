# backend/core/pinn_engine.py
""" Enhanced PINN Engine with Plugin-Based Model Registry
Supports: Solar, Wind, Battery Thermal, Grid Dynamics, Load Forecasting
Features: Runtime model discovery, Physics Validation, Uncertainty Quantification """
import torch
import torch.nn.functional as F
import logging
import os
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Type, Callable
from datetime import datetime
from dataclasses import dataclass, field

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
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            return self.fc(x)

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
    is_bayesian: bool = False

class PhysicsValidator:
    """Validates model outputs against physics constraints"""
    # ... (unchanged from original) ...

class PINNEngine:
    """ Multi-Model Physics-Informed Neural Network Engine with Plugin Registry """

    def __init__(self, enable_gpu: bool = True):
        self.device = torch.device("cuda" if enable_gpu and torch.cuda.is_available() else "cpu")
        
        # Plugin registry: model_name -> ModelPlugin
        self._plugins: Dict[str, "ModelPlugin"] = {}
        self._models: Dict[str, Dict[str, ModelInstance]] = {}  # model_name -> session_id -> ModelInstance
        self._ab_tests: Dict[str, Dict[str, List[str]]] = {}

        self.physics = PhysicsValidator()
        logger.info(f"✅ Enhanced PINN Engine Initialized on {self.device}")

        # Register built-in plugins
        self._register_builtin_plugins()

    def _register_builtin_plugins(self):
        """Register all known model types as plugins"""
        from ml_models.plugins.solar import SolarModelPlugin
        from ml_models.plugins.wind import WindModelPlugin
        from ml_models.plugins.battery_thermal import BatteryThermalModelPlugin
        from ml_models.plugins.grid_voltage import GridVoltageModelPlugin
        from ml_models.plugins.grid_frequency import GridFrequencyModelPlugin
        from ml_models.plugins.load_forecast import LoadForecastModelPlugin

        builtins = [
            SolarModelPlugin(),
            WindModelPlugin(),
            BatteryThermalModelPlugin(),
            GridVoltageModelPlugin(),
            GridFrequencyModelPlugin(),
            LoadForecastModelPlugin(),
        ]
        for plugin in builtins:
            self.register_plugin(plugin)

    def register_plugin(self, plugin: "ModelPlugin"):
        """Register a new model plugin at runtime"""
        if plugin.name in self._plugins:
            logger.warning(f"Plugin {plugin.name} already registered. Overwriting.")
        self._plugins[plugin.name] = plugin
        self._models[plugin.name] = {}
        logger.info(f"🔌 Registered model plugin: {plugin.name}")

    def list_model_types(self) -> List[str]:
        """List all available model types (for diagnostics or UI)"""
        return list(self._plugins.keys())

    def register_model(
        self,
        session_id: str,
        model_name: str,
        version: str = "v1.0",
        artifact_path: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Register a model instance for a session using plugin system"""
        if model_name not in self._plugins:
            logger.error(f"Unknown model type: {model_name}. Available: {list(self._plugins.keys())}")
            return False

        try:
            logger.info(f"🧠 Registering {model_name} model for session {session_id} (v{version})")
            plugin = self._plugins[model_name]
            
            # Build model via plugin
            model = plugin.build_model(**kwargs)
            
            # Load weights if provided
            if artifact_path and os.path.exists(artifact_path):
                state_dict = torch.load(artifact_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"  ✓ Loaded weights from {artifact_path}")
            
            model.to(self.device)
            if kwargs.get("enable_bayesian", False):
                model.train()
            else:
                model.eval()

            model_instance = ModelInstance(
                model=model,
                version=version,
                loaded_at=datetime.now(),
                metrics=ModelMetrics(),
                artifact_path=artifact_path,
                is_bayesian=kwargs.get("enable_bayesian", False)
            )

            self._models[model_name][session_id] = model_instance
            logger.info(f"  ✓ Model registered successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to register {model_name} model: {e}")
            return False

    def setup_ab_test(
        self,
        session_id: str,
        model_name: str,
        version_a: str,
        version_b: str
    ):
        """Configure A/B test between two model versions"""
        if session_id not in self._ab_tests:
            self._ab_tests[session_id] = {}
        self._ab_tests[session_id][model_name] = [version_a, version_b]
        logger.info(f"🧪 A/B test configured: {model_name} ({version_a} vs {version_b})")

    # ========== INFERENCE DELEGATION TO PLUGINS ==========
    def infer_solar_production(
        self,
        session_id: str,
        ghi: float,
        temp: float,
        panel_angle: float,
        cloud_override: Optional[float] = None,
        return_uncertainty: bool = False
    ) -> Tuple[float, Optional[float], float]:
        plugin = self._plugins["solar"]
        input_tensor = plugin.prepare_input(
            ghi=ghi, temp=temp, panel_angle=panel_angle, cloud_override=cloud_override
        )
        output, uncertainty = self._infer(session_id, "solar", input_tensor, return_uncertainty)
        power = output.item()
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
        plugin = self._plugins["wind"]
        input_tensor = plugin.prepare_input(
            wind_speed=wind_speed, air_density=air_density, turbine_state=turbine_state
        )
        output, uncertainty = self._infer(session_id, "wind", input_tensor, return_uncertainty)
        power = output.item()
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
        plugin = self._plugins["battery_thermal"]
        input_tensor = plugin.prepare_input(
            soc=soc, current=current, temp_ambient=temp_ambient, cooling_factor=cooling_factor
        )
        output, uncertainty = self._infer(session_id, "battery_thermal", input_tensor, return_uncertainty)
        temp = output.item()
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
        plugin = self._plugins["grid_voltage"]
        input_tensor = plugin.prepare_input(load=load, generation=generation, topology=topology)
        output, uncertainty = self._infer(session_id, "grid_voltage", input_tensor, return_uncertainty)
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
        plugin = self._plugins["grid_frequency"]
        input_tensor = plugin.prepare_input(power_imbalance=power_imbalance, inertia=inertia)
        output, uncertainty = self._infer(session_id, "grid_frequency", input_tensor, return_uncertainty)
        frequency = output.item()
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
        plugin = self._plugins["load_forecast"]
        input_tensor = plugin.prepare_input(
            historical_window=historical_window, weather=weather, time_features=time_features
        )
        output, uncertainty = self._infer(session_id, "load_forecast", input_tensor, return_uncertainty)
        residual = torch.relu(output - 10.0).sum().item() + torch.relu(-output).sum().item()
        return output, uncertainty, residual

    # ========== CORE INFERENCE ENGINE ==========
    def _infer(
        self,
        session_id: str,
        model_name: str,
        input_tensor: torch.Tensor,
        return_uncertainty: bool = False,
        n_samples: int = 20
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if session_id not in self._models[model_name]:
            logger.warning(f"⚠️ No {model_name} model for session {session_id}, using fallback")
            return self._fallback_prediction(model_name, input_tensor), None

        model_inst = self._models[model_name][session_id]
        model = model_inst.model
        start_time = time.time()

        try:
            input_tensor = input_tensor.to(self.device)
            if not return_uncertainty or not model_inst.is_bayesian:
                with torch.no_grad():
                    output = model(input_tensor).cpu()
                uncertainty = None
            else:
                predictions = []
                for _ in range(n_samples):
                    with torch.no_grad():
                        pred = model(input_tensor).cpu()
                    predictions.append(pred)
                predictions = torch.stack(predictions)
                output = predictions.mean(dim=0)
                uncertainty = predictions.std(dim=0)

            inference_time = time.time() - start_time
            model_inst.metrics.inference_count += 1
            model_inst.metrics.total_inference_time += inference_time
            if uncertainty is not None:
                model_inst.metrics.uncertainty_avg = uncertainty.mean().item()

            return output, uncertainty

        except Exception as e:
            logger.error(f"❌ Inference error for {model_name}: {e}")
            return self._fallback_prediction(model_name, input_tensor), None

    def _fallback_prediction(self, model_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """Generate physics-based fallback when model unavailable"""
        fallbacks = {
            "solar": 0.2,
            "wind": 0.3,
            "battery_thermal": 25.0,
            "grid_voltage": 1.0,
            "grid_frequency": 60.0,
        }
        base = fallbacks.get(model_name, 0.5)
        return torch.ones(input_tensor.shape[0], 1) * base

    # ========== METRICS & DIAGNOSTICS ==========
    def get_model_metrics(self, session_id: str, model_name: str) -> Optional[ModelMetrics]:
        if model_name in self._models and session_id in self._models[model_name]:
            return self._models[model_name][session_id].metrics
        return None

    def get_all_metrics(self, session_id: str) -> Dict[str, ModelMetrics]:
        return {
            name: self.get_model_metrics(session_id, name)
            for name in self._plugins.keys()
            if self.get_model_metrics(session_id, name) is not None
        }

    def log_diagnostics(self, session_id: str):
        logger.info(f"\n🔍 Diagnostics for Session: {session_id}")
        for name in self._plugins.keys():
            metrics = self.get_model_metrics(session_id, name)
            if metrics:
                logger.info(f" {name}:")
                logger.info(f"  Inferences: {metrics.inference_count}")
                logger.info(f"  Avg Time: {metrics.avg_inference_time * 1000:.2f} ms")
                logger.info(f"  Uncertainty: {metrics.uncertainty_avg:.4f}")

# ========== PLUGIN INTERFACE ==========
class ModelPlugin(ABC):
    """Abstract base class for model plugins"""
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier (e.g., 'solar')"""
        pass

    @abstractmethod
    def build_model(self, **kwargs) -> torch.nn.Module:
        """Construct model architecture"""
        pass

    @abstractmethod
    def prepare_input(self, **kwargs) -> torch.Tensor:
        """Convert domain inputs to tensor"""
        pass

# Global instance
pinn_engine = PINNEngine()
