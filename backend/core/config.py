"""
Centralized Configuration Management
Handles all system-wide settings including:
- Database connections (PostgreSQL, TimescaleDB, Redis)
- Model registry and versioning
- Real-time streaming configuration
- Physics constraint thresholds
- Control surface limits
- Monitoring and alerting
- Security and authentication
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from enum import Enum


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """Main configuration class with validation."""
    
    # ========================================================================
    # ENVIRONMENT & GENERAL
    # ========================================================================
    ENV: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )
    DEBUG: bool = Field(default=True, description="Debug mode")
    PROJECT_NAME: str = Field(
        default="Energy Digital Twin",
        description="Project name"
    )
    VERSION: str = Field(default="1.0.0", description="API version")
    API_PREFIX: str = Field(default="/api/v1", description="API route prefix")
    
    # Base directories
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Project root directory"
    )
    
    @property
    def DATA_DIR(self) -> Path:
        """Data storage directory."""
        return self.BASE_DIR / "data"
    
    @property
    def MODELS_DIR(self) -> Path:
        """Model storage directory."""
        return self.BASE_DIR / "models"
    
    @property
    def LOGS_DIR(self) -> Path:
        """Logs directory."""
        return self.BASE_DIR / "logs"
    
    # ========================================================================
    # DATABASE CONFIGURATION
    # ========================================================================
    
    # PostgreSQL (Primary Database)
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_USER: str = Field(default="energy_user", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="energy_pass", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(default="energy_twin", env="POSTGRES_DB")
    
    @property
    def DATABASE_URL(self) -> str:
        """SQLAlchemy database URL."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    # TimescaleDB Configuration (Time-series data)
    TIMESCALE_ENABLED: bool = Field(
        default=True,
        description="Enable TimescaleDB for time-series"
    )
    TIMESCALE_HOST: str = Field(default="localhost", env="TIMESCALE_HOST")
    TIMESCALE_PORT: int = Field(default=5432, env="TIMESCALE_PORT")
    TIMESCALE_USER: str = Field(default="timescale_user", env="TIMESCALE_USER")
    TIMESCALE_PASSWORD: str = Field(
        default="timescale_pass",
        env="TIMESCALE_PASSWORD"
    )
    TIMESCALE_DB: str = Field(default="energy_timeseries", env="TIMESCALE_DB")
    
    @property
    def TIMESCALE_URL(self) -> str:
        """TimescaleDB connection URL."""
        return (
            f"postgresql://{self.TIMESCALE_USER}:{self.TIMESCALE_PASSWORD}"
            f"@{self.TIMESCALE_HOST}:{self.TIMESCALE_PORT}/{self.TIMESCALE_DB}"
        )
    
    # TimescaleDB Settings
    TIMESCALE_CHUNK_INTERVAL: str = Field(
        default="1 day",
        description="Hypertable chunk time interval"
    )
    TIMESCALE_RETENTION_DAYS: int = Field(
        default=90,
        description="Data retention period in days"
    )
    TIMESCALE_COMPRESSION_AFTER: str = Field(
        default="7 days",
        description="Compress chunks older than this"
    )
    
    # Redis Configuration (Caching & Real-time)
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_ENABLED: bool = Field(default=True, description="Enable Redis")
    
    @property
    def REDIS_URL(self) -> str:
        """Redis connection URL."""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Cache TTL settings
    CACHE_TTL_DEFAULT: int = Field(default=300, description="Default cache TTL (seconds)")
    CACHE_TTL_PREDICTIONS: int = Field(default=60, description="Prediction cache TTL")
    CACHE_TTL_METRICS: int = Field(default=30, description="Metrics cache TTL")
    
    # ========================================================================
    # MODEL REGISTRY CONFIGURATION
    # ========================================================================
    
    MODEL_REGISTRY: Dict[str, Dict[str, Any]] = Field(
        default={
            "solar": {
                "name": "Solar PV Predictor",
                "version": "1.0.0",
                "path": "models/solar/solar_pv_v1.pth",
                "type": "pytorch",
                "input_features": ["ghi", "temperature", "cloud_cover"],
                "output": "power_kw",
                "last_trained": None,
                "metrics": {
                    "mae": 0.0,
                    "rmse": 0.0,
                    "r2": 0.0
                }
            },
            "wind": {
                "name": "Wind Turbine Predictor",
                "version": "1.0.0",
                "path": "models/wind/wind_turbine_v1.pth",
                "type": "pytorch",
                "input_features": ["wind_speed", "air_density", "temperature"],
                "output": "power_kw",
                "last_trained": None,
                "metrics": {
                    "mae": 0.0,
                    "rmse": 0.0,
                    "r2": 0.0
                }
            },
            "battery": {
                "name": "Battery Optimizer",
                "version": "1.0.0",
                "path": "models/battery/battery_opt_v1.pth",
                "type": "pytorch",
                "input_features": ["soc", "current", "temperature"],
                "output": "optimal_power_kw",
                "last_trained": None,
                "metrics": {
                    "mae": 0.0,
                    "rmse": 0.0,
                    "r2": 0.0
                }
            },
            "grid": {
                "name": "Grid Demand Predictor",
                "version": "1.0.0",
                "path": "models/grid/grid_demand_v1.pth",
                "type": "pytorch",
                "input_features": ["hour", "temperature", "day_of_week"],
                "output": "demand_kw",
                "last_trained": None,
                "metrics": {
                    "mae": 0.0,
                    "rmse": 0.0,
                    "r2": 0.0
                }
            },
            "ensemble": {
                "name": "Ensemble Meta-Learner",
                "version": "1.0.0",
                "path": "models/ensemble/ensemble_v1.pth",
                "type": "pytorch",
                "input_features": ["all_model_predictions"],
                "output": "combined_prediction",
                "last_trained": None,
                "metrics": {
                    "mae": 0.0,
                    "rmse": 0.0,
                    "r2": 0.0
                }
            },
            "pinn": {
                "name": "Physics-Informed Neural Network",
                "version": "1.0.0",
                "path": "models/pinn/voltage_pinn_v1.pth",
                "type": "pytorch",
                "input_features": ["load_pu", "solar_pu", "fault_magnitude"],
                "output": "voltage_pu",
                "physics_loss_weight": 0.5,
                "last_trained": None,
                "metrics": {
                    "physics_residual": 0.0,
                    "mae": 0.0,
                    "r2": 0.0
                }
            }
        },
        description="Model registry with metadata"
    )
    
    # Model versioning
    MODEL_AUTO_VERSIONING: bool = Field(
        default=True,
        description="Auto-increment version on training"
    )
    MODEL_MAX_VERSIONS: int = Field(
        default=5,
        description="Maximum model versions to keep"
    )
    
    # ========================================================================
    # REAL-TIME STREAMING CONFIGURATION
    # ========================================================================
    
    # WebSocket Settings
    WEBSOCKET_ENABLED: bool = Field(default=True, description="Enable WebSocket")
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(
        default=30,
        description="WebSocket heartbeat interval (seconds)"
    )
    WEBSOCKET_MAX_CONNECTIONS_PER_SESSION: int = Field(
        default=10,
        description="Max concurrent connections per session"
    )
    WEBSOCKET_MESSAGE_QUEUE_SIZE: int = Field(
        default=1000,
        description="Max messages in queue per client"
    )
    
    # Redis Streams Configuration
    REDIS_STREAM_ENABLED: bool = Field(
        default=True,
        description="Enable Redis streams for events"
    )
    REDIS_STREAM_MAX_LEN: int = Field(
        default=10000,
        description="Max stream length per session"
    )
    REDIS_STREAM_RETENTION: int = Field(
        default=3600,
        description="Stream retention in seconds"
    )
    
    # Event types configuration
    EVENT_TYPES: Dict[str, Dict[str, Any]] = Field(
        default={
            "STATE_UPDATE": {
                "priority": "normal",
                "retention": 3600,
                "broadcast": True
            },
            "FAULT_INJECTED": {
                "priority": "high",
                "retention": 7200,
                "broadcast": True
            },
            "CONSTRAINT_VIOLATION": {
                "priority": "critical",
                "retention": 7200,
                "broadcast": True
            },
            "MODEL_PREDICTION": {
                "priority": "normal",
                "retention": 1800,
                "broadcast": False
            },
            "SIMULATION_COMPLETE": {
                "priority": "low",
                "retention": 1800,
                "broadcast": True
            }
        },
        description="Event type configurations"
    )
    
    # ========================================================================
    # CONTROL SURFACE LIMITS
    # ========================================================================
    
    CONTROL_LIMITS: Dict[str, Dict[str, float]] = Field(
        default={
            "battery_power": {
                "min": -100.0,  # kW (discharge)
                "max": 100.0,   # kW (charge)
                "rate_limit": 50.0,  # kW/s max ramp rate
                "description": "Battery charge/discharge power"
            },
            "diesel_power": {
                "min": 0.0,
                "max": 200.0,
                "rate_limit": 20.0,
                "description": "Diesel generator output"
            },
            "curtailment": {
                "min": 0.0,
                "max": 100.0,  # % of available renewable
                "rate_limit": 10.0,
                "description": "Renewable curtailment percentage"
            },
            "v2g_power": {
                "min": 0.0,
                "max": 50.0,
                "rate_limit": 25.0,
                "description": "Vehicle-to-grid discharge power"
            },
            "load_scaling": {
                "min": 0.5,
                "max": 2.0,
                "rate_limit": 0.1,
                "description": "Load demand scaling factor"
            },
            "battery_soc": {
                "min": 0.1,   # 10% minimum
                "max": 0.95,  # 95% maximum
                "critical_low": 0.15,
                "critical_high": 0.90,
                "description": "Battery state of charge"
            },
            "battery_temperature": {
                "min": 0.0,    # °C
                "max": 50.0,   # °C
                "warning": 45.0,
                "critical": 55.0,
                "description": "Battery temperature limits"
            },
            "voltage": {
                "min": 0.95,   # pu
                "max": 1.05,   # pu
                "nominal": 1.0,
                "warning_low": 0.97,
                "warning_high": 1.03,
                "description": "Grid voltage per-unit"
            },
            "frequency": {
                "min": 59.8,   # Hz
                "max": 60.2,   # Hz
                "nominal": 60.0,
                "warning": 0.1,
                "critical": 0.2,
                "description": "Grid frequency"
            }
        },
        description="Control surface and safety limits"
    )
    
    # ========================================================================
    # PHYSICS CONSTRAINT THRESHOLDS
    # ========================================================================
    
    PHYSICS_CONSTRAINTS: Dict[str, Dict[str, Any]] = Field(
        default={
            "power_balance": {
                "threshold": 1e-3,  # 0.1% imbalance
                "severity_multipliers": {
                    "low": 1.0,
                    "medium": 2.0,
                    "high": 5.0,
                    "critical": 10.0
                },
                "description": "Generation-load power balance"
            },
            "voltage_stability": {
                "threshold": 0.05,  # 5% deviation
                "severity_multipliers": {
                    "low": 1.0,
                    "medium": 1.5,
                    "high": 2.0,
                    "critical": 3.0
                },
                "description": "Voltage magnitude stability"
            },
            "frequency_deviation": {
                "threshold": 0.1,   # Hz
                "severity_multipliers": {
                    "low": 1.0,
                    "medium": 2.0,
                    "high": 3.0,
                    "critical": 5.0
                },
                "description": "Frequency deviation from nominal"
            },
            "thermal_violation": {
                "threshold": 1.0,   # °C above limit
                "severity_multipliers": {
                    "low": 1.0,
                    "medium": 2.0,
                    "high": 5.0,
                    "critical": 10.0
                },
                "description": "Battery thermal limits"
            },
            "soc_violation": {
                "threshold": 0.02,  # 2% outside safe range
                "severity_multipliers": {
                    "low": 1.0,
                    "medium": 2.0,
                    "high": 5.0,
                    "critical": 10.0
                },
                "description": "Battery SOC safe operating range"
            },
            "energy_conservation": {
                "threshold": 1e-2,  # 1% energy imbalance
                "severity_multipliers": {
                    "low": 1.0,
                    "medium": 2.0,
                    "high": 5.0,
                    "critical": 10.0
                },
                "description": "Energy conservation law"
            },
            "swing_equation": {
                "threshold": 1e-4,
                "severity_multipliers": {
                    "low": 1.0,
                    "medium": 5.0,
                    "high": 10.0,
                    "critical": 20.0
                },
                "description": "Rotor angle dynamics"
            },
            "composite_loss": {
                "threshold": 1e-4,
                "description": "Overall physics compliance"
            }
        },
        description="Physics constraint validation thresholds"
    )
    
    # Physics loss weights for PINN training
    PHYSICS_LOSS_WEIGHTS: Dict[str, float] = Field(
        default={
            "power_flow": 1.0,
            "swing_equation": 0.5,
            "thermal_dynamics": 0.3,
            "battery_health": 0.2,
            "energy_conservation": 0.8
        },
        description="Loss weights for physics-informed training"
    )
    
    # ========================================================================
    # MONITORING & ALERTING CONFIGURATION
    # ========================================================================
    
    # Monitoring settings
    MONITORING_ENABLED: bool = Field(
        default=True,
        description="Enable monitoring system"
    )
    MONITORING_INTERVAL_SECONDS: int = Field(
        default=10,
        description="Monitoring check interval"
    )
    METRICS_RETENTION_DAYS: int = Field(
        default=30,
        description="Metrics retention period"
    )
    
    # Alert thresholds
    ALERT_THRESHOLDS: Dict[str, Dict[str, Any]] = Field(
        default={
            "cpu_usage": {
                "warning": 70.0,
                "critical": 90.0,
                "unit": "percent"
            },
            "memory_usage": {
                "warning": 75.0,
                "critical": 90.0,
                "unit": "percent"
            },
            "disk_usage": {
                "warning": 80.0,
                "critical": 95.0,
                "unit": "percent"
            },
            "response_time": {
                "warning": 1000.0,  # ms
                "critical": 5000.0,
                "unit": "milliseconds"
            },
            "error_rate": {
                "warning": 0.01,    # 1%
                "critical": 0.05,   # 5%
                "unit": "ratio"
            },
            "prediction_accuracy": {
                "warning": 0.85,    # Below 85% R²
                "critical": 0.70,   # Below 70% R²
                "unit": "r_squared"
            },
            "physics_violation_rate": {
                "warning": 0.05,    # 5% of predictions
                "critical": 0.20,   # 20% of predictions
                "unit": "ratio"
            }
        },
        description="Alerting thresholds for monitoring"
    )
    
    # Alert channels
    ALERT_CHANNELS: Dict[str, bool] = Field(
        default={
            "email": False,
            "slack": False,
            "webhook": True,
            "log": True
        },
        description="Enabled alert channels"
    )
    
    # Webhook configuration
    ALERT_WEBHOOK_URL: Optional[str] = Field(
        default=None,
        env="ALERT_WEBHOOK_URL",
        description="Webhook URL for alerts"
    )
    
    # Email configuration
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USER: Optional[str] = Field(default=None, env="SMTP_USER")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    ALERT_EMAIL_FROM: str = Field(
        default="alerts@energytwin.com",
        env="ALERT_EMAIL_FROM"
    )
    ALERT_EMAIL_TO: Optional[str] = Field(default=None, env="ALERT_EMAIL_TO")
    
    # ========================================================================
    # SIMULATION CONFIGURATION
    # ========================================================================
    
    SIMULATION_DEFAULT_INTERVAL_MS: int = Field(
        default=1000,
        description="Default simulation step interval (ms)"
    )
    SIMULATION_MAX_TICKS: int = Field(
        default=8760,  # One year at hourly resolution
        description="Maximum simulation ticks"
    )
    SIMULATION_PARALLEL_MODELS: bool = Field(
        default=True,
        description="Execute models in parallel"
    )
    
    # ========================================================================
    # SECURITY & AUTHENTICATION
    # ========================================================================
    
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY",
        description="JWT secret key"
    )
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60,
        description="Access token expiration"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        description="Refresh token expiration"
    )
    
    # CORS settings
    CORS_ORIGINS: list = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: list = Field(default=["*"])
    CORS_ALLOW_HEADERS: list = Field(default=["*"])
    
    # ========================================================================
    # LOGGING CONFIGURATION
    # ========================================================================
    
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    LOG_TO_FILE: bool = Field(default=True, description="Enable file logging")
    LOG_FILE_MAX_BYTES: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Max log file size"
    )
    LOG_FILE_BACKUP_COUNT: int = Field(
        default=5,
        description="Number of log file backups"
    )
    
    # ========================================================================
    # VALIDATORS
    # ========================================================================
    
    @validator("MODELS_DIR", "DATA_DIR", "LOGS_DIR", pre=False, always=True)
    def create_directories(cls, v):
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

settings = Settings()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path(model_name: str, version: Optional[str] = None) -> Path:
    """Get absolute path to model file."""
    model_config = settings.MODEL_REGISTRY.get(model_name)
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    model_path = settings.BASE_DIR / model_config["path"]
    
    if version:
        # Use specific version
        model_path = model_path.parent / f"{model_path.stem}_v{version}{model_path.suffix}"
    
    return model_path


def get_control_limit(parameter: str, limit_type: str = "max") -> float:
    """Get control limit for a parameter."""
    param_config = settings.CONTROL_LIMITS.get(parameter)
    if not param_config:
        raise ValueError(f"Parameter '{parameter}' not found in control limits")
    
    return param_config.get(limit_type, 0.0)


def get_physics_threshold(constraint: str) -> float:
    """Get physics constraint threshold."""
    constraint_config = settings.PHYSICS_CONSTRAINTS.get(constraint)
    if not constraint_config:
        raise ValueError(f"Constraint '{constraint}' not found in physics constraints")
    
    return constraint_config["threshold"]


def is_critical_violation(constraint: str, residual: float) -> bool:
    """Check if residual exceeds critical threshold."""
    constraint_config = settings.PHYSICS_CONSTRAINTS.get(constraint)
    if not constraint_config:
        return False
    
    threshold = constraint_config["threshold"]
    critical_multiplier = constraint_config["severity_multipliers"]["critical"]
    
    return residual > (threshold * critical_multiplier)


def get_database_url(use_timescale: bool = False) -> str:
    """Get appropriate database URL."""
    if use_timescale and settings.TIMESCALE_ENABLED:
        return settings.TIMESCALE_URL
    return settings.DATABASE_URL


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_settings():
    """Initialize settings and create necessary directories."""
    # Create all required directories
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create model subdirectories
    for model_name in settings.MODEL_REGISTRY.keys():
        model_dir = settings.MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Configuration initialized for {settings.ENV.value} environment")
    print(f"📁 Base directory: {settings.BASE_DIR}")
    print(f"📊 Database: {settings.DATABASE_URL.split('@')[1]}")  # Hide credentials
    if settings.TIMESCALE_ENABLED:
        print(f"⏰ TimescaleDB: {settings.TIMESCALE_URL.split('@')[1]}")
    if settings.REDIS_ENABLED:
        print(f"🔴 Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    print(f"🧠 Models directory: {settings.MODELS_DIR}")


# Auto-initialize on import
if __name__ != "__main__":
    initialize_settings()

