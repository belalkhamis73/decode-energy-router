```python
"""
Enhanced NREL Data Pipeline with Multi-Source Fusion & Physics-Aware Augmentation.
Compliant with SaaS Digital Twin MLOps Standards.
"""

import logging
import requests
import pandas as pd
import numpy as np
import torch
import pvlib
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from enum import Enum
from datetime import datetime, timedelta

# Configure logging as per Technical Blueprint
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants (Should ideally be imported from physics-core)
SOLAR_CONSTANT_GHI = 1361.0  # W/m^2
MAX_CLEARNESS_INDEX = 1.0
MIN_TEMPERATURE = -50.0  # Celsius
MAX_TEMPERATURE = 60.0
MAX_WIND_SPEED = 50.0  # m/s

class DataSource(Enum):
    """Enumeration of supported data sources."""
    NREL = "nrel"
    SCADA = "scada"
    USER_UPLOAD = "user_upload"
    SYNTHETIC = "synthetic"

class ModelType(Enum):
    """Specialized model types requiring different features."""
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly"
    DEGRADATION = "degradation"
    OPTIMIZATION = "optimization"

@dataclass(frozen=True)
class GeoLocation:
    """Immutable value object for site coordinates."""
    lat: float
    lon: float

@dataclass
class DataQualityMetrics:
    """Metrics for data quality assessment."""
    completeness: float  # 0-1
    validity: float  # 0-1
    consistency: float  # 0-1
    timeliness: float  # 0-1
    anomaly_score: float  # 0-1
    overall_score: float  # 0-1
    
    def __post_init__(self):
        self.overall_score = np.mean([
            self.completeness, self.validity, 
            self.consistency, self.timeliness
        ]) * (1 - self.anomaly_score)

class NRELClient:
    """
    Handles communication with the NREL API.
    Principle: Single Responsibility (Data Fetching).
    """
    BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is required for NREL Client.")
        self._api_key = api_key

    def fetch_data(self, location: GeoLocation, year: str, email: str) -> pd.DataFrame:
        """
        Fetches hourly solar data for a specific location and year.
        Implements 'Fail Fast' by raising HTTP errors immediately.
        """
        params = {
            "api_key": self._api_key,
            "wkt": f"POINT({location.lon} {location.lat})",
            "names": year,
            "interval": "60",
            "attributes": "ghi,dni,dhi,air_temperature,wind_speed",
            "utc": "true",
            "email": email
        }

        try:
            logger.info(f"Fetching NREL data for location: {location}")
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            filename = f"nrel_raw_{year}.csv"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            df = pd.read_csv(filename, skiprows=2)
            df['source'] = DataSource.NREL.value
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"NREL API Request failed: {e}")
            raise

class MultiSourceDataFusion:
    """
    Combines data from multiple sources (NREL, SCADA, user uploads).
    Handles timestamp alignment, unit conversion, and quality-weighted merging.
    """
    
    def __init__(self, location: GeoLocation):
        self.location = location
        self.source_weights = {
            DataSource.SCADA: 1.0,  # Highest priority
            DataSource.USER_UPLOAD: 0.8,
            DataSource.NREL: 0.6,
            DataSource.SYNTHETIC: 0.3
        }
    
    def standardize_columns(self, df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
        """Harmonize column names across different sources."""
        column_mapping = {
            DataSource.NREL: {
                'GHI': 'ghi', 'DNI': 'dni', 'DHI': 'dhi',
                'Temperature': 'temp', 'Wind Speed': 'wind_speed'
            },
            DataSource.SCADA: {
                'global_irradiance': 'ghi', 'direct_irradiance': 'dni',
                'diffuse_irradiance': 'dhi', 'ambient_temp': 'temp',
                'wind_vel': 'wind_speed', 'power_output': 'power'
            },
            DataSource.USER_UPLOAD: {
                'irradiance': 'ghi', 'temperature': 'temp', 'wind': 'wind_speed'
            }
        }
        
        if source in column_mapping:
            df = df.rename(columns=column_mapping[source])
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns and 'time_index' in df.columns:
            df['timestamp'] = df['time_index']
        elif 'timestamp' not in df.columns and all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
            df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        
        df['source'] = source.value
        return df
    
    def merge_sources(self, data_dict: Dict[DataSource, pd.DataFrame], 
                      quality_scores: Optional[Dict[DataSource, float]] = None) -> pd.DataFrame:
        """
        Quality-weighted merging of multiple data sources.
        Prioritizes higher quality sources for overlapping timestamps.
        """
        if not data_dict:
            raise ValueError("No data sources provided for merging")
        
        # Standardize all sources
        standardized = {}
        for source, df in data_dict.items():
            std_df = self.standardize_columns(df.copy(), source)
            if 'timestamp' in std_df.columns:
                std_df = std_df.set_index('timestamp').sort_index()
            standardized[source] = std_df
        
        # Calculate effective weights (base weight * quality score)
        effective_weights = {}
        for source in standardized.keys():
            base_weight = self.source_weights.get(source, 0.5)
            quality = quality_scores.get(source, 1.0) if quality_scores else 1.0
            effective_weights[source] = base_weight * quality
        
        # Merge with priority (weighted average for overlaps)
        all_columns = set()
        for df in standardized.values():
            all_columns.update(df.columns)
        all_columns.discard('source')
        
        # Get union of all timestamps
        all_timestamps = pd.Index([])
        for df in standardized.values():
            all_timestamps = all_timestamps.union(df.index)
        
        # Initialize merged dataframe
        merged = pd.DataFrame(index=all_timestamps)
        
        for col in all_columns:
            weighted_sum = pd.Series(0.0, index=all_timestamps)
            weight_sum = pd.Series(0.0, index=all_timestamps)
            
            for source, df in standardized.items():
                if col in df.columns:
                    mask = df[col].notna()
                    weight = effective_weights[source]
                    weighted_sum[mask] += df.loc[mask, col] * weight
                    weight_sum[mask] += weight
            
            merged[col] = weighted_sum / weight_sum.replace(0, np.nan)
        
        merged = merged.reset_index().rename(columns={'index': 'timestamp'})
        logger.info(f"Merged {len(data_dict)} sources into {len(merged)} records")
        return merged

class DataQualityMonitor:
    """
    Validates incoming data streams with physics-based checks.
    Scores data quality and detects anomalies.
    """
    
    def __init__(self, location: GeoLocation):
        self.location = location
    
    def check_completeness(self, df: pd.DataFrame, required_cols: List[str]) -> float:
        """Calculate completeness ratio."""
        if df.empty:
            return 0.0
        missing_ratio = df[required_cols].isna().sum().sum() / (len(df) * len(required_cols))
        return 1.0 - missing_ratio
    
    def check_validity(self, df: pd.DataFrame) -> float:
        """Check physical validity of values."""
        violations = 0
        total_checks = 0
        
        checks = [
            ('ghi', lambda x: (x >= 0) & (x <= SOLAR_CONSTANT_GHI * 1.2)),
            ('dni', lambda x: (x >= 0) & (x <= SOLAR_CONSTANT_GHI * 1.2)),
            ('dhi', lambda x: (x >= 0) & (x <= SOLAR_CONSTANT_GHI)),
            ('temp', lambda x: (x >= MIN_TEMPERATURE) & (x <= MAX_TEMPERATURE)),
            ('wind_speed', lambda x: (x >= 0) & (x <= MAX_WIND_SPEED))
        ]
        
        for col, check_fn in checks:
            if col in df.columns:
                valid = check_fn(df[col])
                violations += (~valid).sum()
                total_checks += len(df)
        
        return 1.0 - (violations / max(total_checks, 1))
    
    def check_consistency(self, df: pd.DataFrame) -> float:
        """Check physical consistency (GHI ≈ DNI*cos(θ) + DHI)."""
        if not all(col in df.columns for col in ['ghi', 'dni', 'dhi']):
            return 1.0
        
        # Calculate expected GHI from components (need solar position)
        if 'timestamp' in df.columns:
            df_temp = df.set_index('timestamp')
            site = pvlib.location.Location(self.location.lat, self.location.lon, tz='UTC')
            solar_pos = site.get_solarposition(df_temp.index)
            cos_zenith = np.cos(np.radians(solar_pos['zenith'])).clip(lower=0)
            
            expected_ghi = df['dni'] * cos_zenith.values + df['dhi']
            consistency_error = np.abs(df['ghi'] - expected_ghi) / (df['ghi'] + 1e-6)
            consistency_score = 1.0 - consistency_error.clip(0, 1).mean()
            return consistency_score
        
        return 1.0
    
    def check_timeliness(self, df: pd.DataFrame, max_delay_hours: int = 24) -> float:
        """Check data freshness."""
        if 'timestamp' not in df.columns or df.empty:
            return 0.5
        
        latest_timestamp = pd.to_datetime(df['timestamp']).max()
        delay = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        if delay <= max_delay_hours:
            return 1.0
        elif delay <= max_delay_hours * 7:
            return 1.0 - (delay - max_delay_hours) / (max_delay_hours * 6)
        else:
            return 0.0
    
    def detect_anomalies(self, df: pd.DataFrame) -> Tuple[pd.Series, float]:
        """Simple statistical anomaly detection."""
        anomaly_mask = pd.Series(False, index=df.index)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns and df[col].notna().sum() > 10:
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / (std + 1e-6))
                anomaly_mask |= (z_scores > 4)  # 4-sigma outliers
        
        anomaly_ratio = anomaly_mask.sum() / len(df) if len(df) > 0 else 0.0
        return anomaly_mask, anomaly_ratio
    
    def assess_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Comprehensive data quality assessment."""
        required_cols = ['ghi', 'dni', 'dhi', 'temp', 'wind_speed']
        available_cols = [col for col in required_cols if col in df.columns]
        
        completeness = self.check_completeness(df, available_cols)
        validity = self.check_validity(df)
        consistency = self.check_consistency(df)
        timeliness = self.check_timeliness(df)
        _, anomaly_score = self.detect_anomalies(df)
        
        metrics = DataQualityMetrics(
            completeness=completeness,
            validity=validity,
            consistency=consistency,
            timeliness=timeliness,
            anomaly_score=anomaly_score,
            overall_score=0.0  # Will be calculated in __post_init__
        )
        
        logger.info(f"Quality Assessment - Overall: {metrics.overall_score:.2f}, "
                   f"Completeness: {completeness:.2f}, Validity: {validity:.2f}")
        return metrics

class PhysicsAwareAugmentation:
    """
    Generates synthetic data respecting physical laws.
    Uses parametric perturbations with physics constraints.
    """
    
    def __init__(self, location: GeoLocation, seed: int = 42):
        self.location = location
        self.rng = np.random.RandomState(seed)
    
    def generate_cloud_transients(self, df: pd.DataFrame, 
                                  n_events: int = 10,
                                  duration_range: Tuple[int, int] = (5, 30)) -> pd.DataFrame:
        """
        Generate synthetic cloud passing events.
        Maintains GHI = DNI*cos(θ) + DHI relationship.
        """
        augmented = df.copy()
        
        for _ in range(n_events):
            # Random event start
            start_idx = self.rng.randint(0, len(df) - max(duration_range))
            duration = self.rng.randint(*duration_range)
            
            # Cloud attenuation (0.3 to 0.9)
            attenuation = self.rng.uniform(0.3, 0.9)
            
            # Apply to DNI primarily (clouds block direct radiation)
            if 'dni' in augmented.columns:
                original_dni = augmented.loc[start_idx:start_idx+duration, 'dni'].copy()
                augmented.loc[start_idx:start_idx+duration, 'dni'] *= attenuation
                
                # Increase DHI proportionally (scattered light)
                if 'dhi' in augmented.columns:
                    scattered = original_dni * (1 - attenuation) * 0.2
                    augmented.loc[start_idx:start_idx+duration, 'dhi'] += scattered
                
                # Recalculate GHI to maintain consistency
                if all(col in augmented.columns for col in ['ghi', 'dni', 'dhi', 'timestamp']):
                    self._recalculate_ghi(augmented, start_idx, start_idx+duration)
        
        logger.info(f"Generated {n_events} cloud transient events")
        return augmented
    
    def generate_soiling_degradation(self, df: pd.DataFrame, 
                                     degradation_rate: float = 0.005) -> pd.DataFrame:
        """
        Simulate gradual soiling effects on irradiance measurements.
        Linear degradation over time.
        """
        augmented = df.copy()
        
        if 'timestamp' in augmented.columns:
            augmented['timestamp'] = pd.to_datetime(augmented['timestamp'])
            time_delta = (augmented['timestamp'] - augmented['timestamp'].min()).dt.total_seconds()
            days_elapsed = time_delta / 86400
            
            # Gradual reduction factor
            reduction = 1.0 - (degradation_rate * days_elapsed / 365)
            reduction = reduction.clip(lower=0.7)  # Max 30% degradation
            
            for col in ['ghi', 'dni', 'dhi']:
                if col in augmented.columns:
                    augmented[col] *= reduction.values
        
        logger.info(f"Applied soiling degradation (rate={degradation_rate})")
        return augmented
    
    def generate_rare_weather(self, df: pd.DataFrame, event_type: str = 'storm') -> pd.DataFrame:
        """
        Generate rare weather events (storms, heatwaves, extreme wind).
        """
        augmented = df.copy()
        
        if event_type == 'storm':
            # High wind, low irradiance, temperature drop
            if len(augmented) > 50:
                start = self.rng.randint(0, len(augmented) - 50)
                augmented.loc[start:start+50, 'wind_speed'] = self.rng.uniform(15, 25, 51)
                augmented.loc[start:start+50, 'ghi'] *= self.rng.uniform(0.1, 0.3)
                if 'temp' in augmented.columns:
                    augmented.loc[start:start+50, 'temp'] -= self.rng.uniform(5, 10)
        
        elif event_type == 'heatwave':
            # High temperature, clear sky
            if len(augmented) > 100:
                start = self.rng.randint(0, len(augmented) - 100)
                if 'temp' in augmented.columns:
                    augmented.loc[start:start+100, 'temp'] += self.rng.uniform(10, 20)
        
        logger.info(f"Generated rare weather event: {event_type}")
        return augmented
    
    def _recalculate_ghi(self, df: pd.DataFrame, start_idx: int, end_idx: int):
        """Recalculate GHI from DNI and DHI to maintain physics consistency."""
        subset = df.loc[start_idx:end_idx].copy()
        subset['timestamp'] = pd.to_datetime(subset['timestamp'])
        subset = subset.set_index('timestamp')
        
        site = pvlib.location.Location(self.location.lat, self.location.lon, tz='UTC')
        solar_pos = site.get_solarposition(subset.index)
        cos_zenith = np.cos(np.radians(solar_pos['zenith'])).clip(lower=0)
        
        df.loc[start_idx:end_idx, 'ghi'] = (
            df.loc[start_idx:end_idx, 'dni'] * cos_zenith.values +
            df.loc[start_idx:end_idx, 'dhi']
        )

class DomainSpecificFeatureEngineer:
    """
    Creates specialized features for different model types.
    Each model type gets optimized feature sets.
    """
    
    def __init__(self, location: GeoLocation):
        self.location = location
    
    def engineer_forecasting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for time-series forecasting models."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['month'] = df['timestamp'].dt.month
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            # Lag features
            for col in ['ghi', 'temp', 'wind_speed']:
                if col in df.columns:
                    df[f'{col}_lag1'] = df[col].shift(1)
                    df[f'{col}_lag24'] = df[col].shift(24)
                    df[f'{col}_rolling_mean_6'] = df[col].rolling(6, min_periods=1).mean()
        
        logger.info(f"Engineered {df.shape[1]} forecasting features")
        return df
    
    def engineer_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for anomaly detection models."""
        df = df.copy()
        
        # Deviation from expected patterns
        for col in ['ghi', 'temp', 'wind_speed']:
            if col in df.columns:
                rolling_mean = df[col].rolling(24, min_periods=1).mean()
                rolling_std = df[col].rolling(24, min_periods=1).std()
                df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-6)
                df[f'{col}_deviation'] = np.abs(df[col] - rolling_mean)
        
        # Rate of change
        for col in ['ghi', 'dni', 'dhi']:
            if col in df.columns:
                df[f'{col}_rate_of_change'] = df[col].diff()
        
        logger.info(f"Engineered {df.shape[1]} anomaly detection features")
        return df
    
    def engineer_degradation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for degradation analysis."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 86400
            # Performance ratio trend
            if all(col in df.columns for col in ['ghi', 'power']):
                df['performance_ratio'] = df['power'] / (df['ghi'] + 1e-6)
                df['pr_rolling_mean_30d'] = df['performance_ratio'].rolling(30*24, min_periods=1).mean()
        
                
        # Clearness index (atmospheric clarity)
        if 'ghi' in df.columns:
            df['clearness_index'] = (df['ghi'] / (SOLAR_CONSTANT_GHI * 0.8)).clip(0, 1)
        
        logger.info(f"Engineered {df.shape[1]} degradation features")
        return df
    
    def engineer_optimization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for operational optimization."""
        df = df.copy()
        
        # Energy yield potential
        if 'ghi' in df.columns:
            df['daily_ghi_sum'] = df.groupby(df['timestamp'].dt.date)['ghi'].transform('sum')
        
        # Optimal tilt angle indicators
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_temp = df.set_index('timestamp')
            site = pvlib.location.Location(self.location.lat, self.location.lon, tz='UTC')
            solar_pos = site.get_solarposition(df_temp.index)
            df['solar_elevation'] = solar_pos['apparent_elevation'].values
            df['solar_azimuth'] = solar_pos['azimuth'].values
        
        logger.info(f"Engineered {df.shape[1]} optimization features")
        return df
    
    def get_features_for_model(self, df: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
        """Route to appropriate feature engineering based on model type."""
        feature_map = {
            ModelType.FORECASTING: self.engineer_forecasting_features,
            ModelType.ANOMALY_DETECTION: self.engineer_anomaly_features,
            ModelType.DEGRADATION: self.engineer_degradation_features,
            ModelType.OPTIMIZATION: self.engineer_optimization_features
        }
        
        if model_type in feature_map:
            return feature_map[model_type](df)
        else:
            logger.warning(f"Unknown model type: {model_type}, returning original dataframe")
            return df

class PhysicsProcessor:
    """
    Enforces physical constraints on raw solar data.
    Principle: Domain Logic Isolation.
    """
    
    @staticmethod
    def enforce_solar_geometry(df: pd.DataFrame, location: GeoLocation) -> pd.DataFrame:
        """
        Applies a 'Night Mask' based on astronomical geometry.
        Ensures irradiance is exactly zero when the sun is below the horizon.
        """
        if 'timestamp' not in df.columns:
            if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
                df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            elif 'time_index' in df.columns:
                df['timestamp'] = df['time_index']
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_indexed = df.set_index('timestamp')
        
        site = pvlib.location.Location(location.lat, location.lon, tz='UTC')
        solar_position = site.get_solarposition(df_indexed.index)
        
        is_night = solar_position['apparent_elevation'] <= 0
        
        cols_to_fix = ['ghi', 'dni', 'dhi', 'GHI', 'DNI', 'DHI']
        for col in cols_to_fix:
            if col in df.columns:
                df.loc[df[col] < 0, col] = 0.0
                df.loc[df.index[is_night], col] = 0.0
        
        # Feature Engineering: Cosine of Zenith Angle
        df['cos_zenith'] = np.cos(np.radians(solar_position['zenith'])).clip(lower=0.0).values
        
        # Enforce causality: GHI should not exceed extraterrestrial radiation
        if 'ghi' in df.columns:
            etr = pvlib.irradiance.get_extra_radiation(df_indexed.index).values
            df['ghi'] = df['ghi'].clip(upper=etr * 1.2)  # 20% margin for measurement uncertainty
        
        logger.info(f"Physics constraints applied. Masked {is_night.sum()} night-time rows.")
        return df.reset_index(drop=True)

class PhysicsScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler that preserves physical relationships between GHI, DNI, and DHI.
    Principle: Open/Closed (Extends BaseEstimator).
    """
    def __init__(self):
        self.temp_mean = 0.0
        self.temp_std = 1.0
        self.wind_max = 1.0

    def fit(self, df: pd.DataFrame):
        if 'temp' in df.columns:
            self.temp_mean = df['temp'].mean()
            self.temp_std = df['temp'].std()
        elif 'Temperature' in df.columns:
            self.temp_mean = df['Temperature'].mean()
            self.temp_std = df['Temperature'].std()
            
        if 'wind_speed' in df.columns:
            self.wind_max = df['wind_speed'].max()
        elif 'Wind Speed' in df.columns:
            self.wind_max = df['Wind Speed'].max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()
        
        # Scale Irradiance by Solar Constant
        irr_cols = ['ghi', 'dni', 'dhi', 'GHI', 'DNI', 'DHI']
        for col in irr_cols:
            if col in df_scaled.columns:
                df_scaled[col] = df_scaled[col] / SOLAR_CONSTANT_GHI
        
        # Standard Scaling for Temperature
        temp_cols = ['temp', 'Temperature']
        for col in temp_cols:
            if col in df_scaled.columns:
                df_scaled[col] = (df_scaled[col] - self.temp_mean) / (self.temp_std + 1e-6)
        
        # MinMax Scaling for Wind
        wind_cols = ['wind_speed', 'Wind Speed']
        for col in wind_cols:
            if col in df_scaled.columns:
                df_scaled[col] = df_scaled[col] / (self.wind_max + 1e-6)
        
        return df_scaled

class SolarDataset(Dataset):
    """
    PyTorch Dataset for LSTM-PINN training.
    Principle: Interface Segregation (Complies with torch.utils.data.Dataset).
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 24, target_col: str = 'ghi'):
        self.seq_len = seq_len
        self.feature_cols = ['dni', 'dhi', 'temp', 'wind_speed', 'cos_zenith']
        
        # Fallback to capitalized versions if needed
        actual_cols = []
        for col in self.feature_cols:
            if col in df.columns:
                actual_cols.append(col)
            elif col.upper() in df.columns:
                actual_cols.append(col.upper())
            elif col.title() in df.columns:
                actual_cols.append(col.title())
        
        self.feature_cols = actual_cols
        
        if not self.feature_cols:
            raise ValueError(f"DataFrame missing required feature columns")

        self.features = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        
        # Handle target column
        target_actual = target_col if target_col in df.columns else target_col.upper()
        self.target = torch.tensor(df[[target_actual]].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.target[idx + self.seq_len]
        physics_input = self.features[idx + self.seq_len, -1]
        
        return x, y, physics_input

# --- Usage Example (to be triggered by main orchestration script) ---
if __name__ == "__main__":
    # Configuration (In production, load from environment variables)
    LOC = GeoLocation(lat=31.2357, lon=30.0444)
    API_KEY = "DEMO_KEY"  # Replace with valid key
    
    try:
        logger.info("=" * 80)
        logger.info("ENHANCED DATA PIPELINE - MULTI-SOURCE FUSION & AUGMENTATION")
        logger.info("=" * 80)
        
        # ===== STEP 1: Multi-Source Data Collection =====
        logger.info("\n[STEP 1] Fetching data from multiple sources...")
        
        # Simulate NREL data
        nrel_client = NRELClient(API_KEY)
        try:
            nrel_df = nrel_client.fetch_data(LOC, "2022", "user@example.com")
            logger.info(f"✓ NREL data fetched: {len(nrel_df)} records")
        except Exception as e:
            logger.warning(f"NREL fetch failed (using mock data): {e}")
            # Create mock NREL data
            dates = pd.date_range('2022-01-01', '2022-01-31', freq='H')
            nrel_df = pd.DataFrame({
                'Year': dates.year,
                'Month': dates.month,
                'Day': dates.day,
                'Hour': dates.hour,
                'Minute': dates.minute,
                'GHI': np.random.uniform(0, 800, len(dates)),
                'DNI': np.random.uniform(0, 900, len(dates)),
                'DHI': np.random.uniform(0, 300, len(dates)),
                'Temperature': np.random.uniform(15, 30, len(dates)),
                'Wind Speed': np.random.uniform(0, 10, len(dates)),
            })
        
        # Simulate SCADA data (typically higher quality, real-time)
        dates_scada = pd.date_range('2022-01-15', '2022-01-31', freq='H')
        scada_df = pd.DataFrame({
            'timestamp': dates_scada,
            'global_irradiance': np.random.uniform(0, 850, len(dates_scada)),
            'direct_irradiance': np.random.uniform(0, 920, len(dates_scada)),
            'diffuse_irradiance': np.random.uniform(0, 320, len(dates_scada)),
            'ambient_temp': np.random.uniform(16, 32, len(dates_scada)),
            'wind_vel': np.random.uniform(0, 12, len(dates_scada)),
            'power_output': np.random.uniform(0, 500, len(dates_scada))
        })
        logger.info(f"✓ SCADA data simulated: {len(scada_df)} records")
        
        # Simulate user upload data (partial coverage, specific periods)
        dates_user = pd.date_range('2022-01-20', '2022-01-25', freq='H')
        user_df = pd.DataFrame({
            'timestamp': dates_user,
            'irradiance': np.random.uniform(0, 800, len(dates_user)),
            'temperature': np.random.uniform(18, 28, len(dates_user)),
            'wind': np.random.uniform(0, 8, len(dates_user))
        })
        logger.info(f"✓ User upload data simulated: {len(user_df)} records")
        
        # ===== STEP 2: Data Quality Assessment =====
        logger.info("\n[STEP 2] Assessing data quality...")
        quality_monitor = DataQualityMonitor(LOC)
        
        # Assess each source
        quality_scores = {}
        
        # Standardize NREL first for quality check
        fusion = MultiSourceDataFusion(LOC)
        nrel_std = fusion.standardize_columns(nrel_df.copy(), DataSource.NREL)
        scada_std = fusion.standardize_columns(scada_df.copy(), DataSource.SCADA)
        user_std = fusion.standardize_columns(user_df.copy(), DataSource.USER_UPLOAD)
        
        for name, df, source in [
            ("NREL", nrel_std, DataSource.NREL),
            ("SCADA", scada_std, DataSource.SCADA),
            ("User Upload", user_std, DataSource.USER_UPLOAD)
        ]:
            metrics = quality_monitor.assess_quality(df)
            quality_scores[source] = metrics.overall_score
            logger.info(f"  {name}: Overall Score = {metrics.overall_score:.3f} "
                       f"(C:{metrics.completeness:.2f}, V:{metrics.validity:.2f}, "
                       f"CS:{metrics.consistency:.2f}, A:{metrics.anomaly_score:.2f})")
        
        # ===== STEP 3: Multi-Source Fusion =====
        logger.info("\n[STEP 3] Fusing multiple data sources...")
        
        data_sources = {
            DataSource.NREL: nrel_df,
            DataSource.SCADA: scada_df,
            DataSource.USER_UPLOAD: user_df
        }
        
        merged_df = fusion.merge_sources(data_sources, quality_scores)
        logger.info(f"✓ Merged dataset: {len(merged_df)} records, {merged_df.shape[1]} features")
        
        # ===== STEP 4: Physics-Based Processing =====
        logger.info("\n[STEP 4] Applying physics-based constraints...")
        processor = PhysicsProcessor()
        clean_df = processor.enforce_solar_geometry(merged_df, LOC)
        logger.info(f"✓ Physics constraints enforced")
        
        # ===== STEP 5: Data Augmentation (Physics-Aware) =====
        logger.info("\n[STEP 5] Generating physics-aware synthetic data...")
        augmenter = PhysicsAwareAugmentation(LOC, seed=42)
        
        # Create augmented versions for training robustness
        aug_cloud = augmenter.generate_cloud_transients(clean_df.copy(), n_events=5)
        aug_soiling = augmenter.generate_soiling_degradation(clean_df.copy(), degradation_rate=0.003)
        aug_storm = augmenter.generate_rare_weather(clean_df.copy(), event_type='storm')
        aug_heatwave = augmenter.generate_rare_weather(clean_df.copy(), event_type='heatwave')
        
        # Combine original + augmented data
        all_data = pd.concat([clean_df, aug_cloud, aug_soiling, aug_storm, aug_heatwave], 
                            ignore_index=True)
        logger.info(f"✓ Augmented dataset: {len(all_data)} records (original + synthetic)")
        
        # ===== STEP 6: Domain-Specific Feature Engineering =====
        logger.info("\n[STEP 6] Engineering domain-specific features...")
        engineer = DomainSpecificFeatureEngineer(LOC)
        
        # Create features for different model types
        forecasting_features = engineer.get_features_for_model(
            clean_df.copy(), ModelType.FORECASTING
        )
        logger.info(f"  Forecasting: {forecasting_features.shape[1]} features")
        
        anomaly_features = engineer.get_features_for_model(
            clean_df.copy(), ModelType.ANOMALY_DETECTION
        )
        logger.info(f"  Anomaly Detection: {anomaly_features.shape[1]} features")
        
        degradation_features = engineer.get_features_for_model(
            clean_df.copy(), ModelType.DEGRADATION
        )
        logger.info(f"  Degradation: {degradation_features.shape[1]} features")
        
        optimization_features = engineer.get_features_for_model(
            clean_df.copy(), ModelType.OPTIMIZATION
        )
        logger.info(f"  Optimization: {optimization_features.shape[1]} features")
        
        # ===== STEP 7: Normalization & Dataset Creation =====
        logger.info("\n[STEP 7] Normalizing and creating PyTorch datasets...")
        scaler = PhysicsScaler()
        
        # Use forecasting features as the main dataset
        norm_df = scaler.fit(forecasting_features).transform(forecasting_features)
        
        # Drop rows with NaN values from lag/rolling features
        norm_df = norm_df.dropna()
        logger.info(f"✓ Normalized dataset: {len(norm_df)} records after cleaning")
        
        # Create PyTorch dataset
        dataset = SolarDataset(norm_df, seq_len=24, target_col='ghi')
        logger.info(f"✓ PyTorch Dataset created: {len(dataset)} sequences")
        
        # Sample a batch to verify
        if len(dataset) > 0:
            sample_x, sample_y, sample_physics = dataset[0]
            logger.info(f"  Sample shapes - X: {sample_x.shape}, Y: {sample_y.shape}, "
                       f"Physics: {sample_physics.shape}")
        
        # ===== STEP 8: Quality Report =====
        logger.info("\n[STEP 8] Final quality report...")
        final_metrics = quality_monitor.assess_quality(norm_df)
        
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Records Processed: {len(all_data):,}")
        logger.info(f"Training Sequences Available: {len(dataset):,}")
        logger.info(f"Final Quality Score: {final_metrics.overall_score:.3f}")
        logger.info(f"Features Engineered: {norm_df.shape[1]}")
        logger.info(f"Data Sources Used: {len(data_sources)}")
        logger.info(f"Augmentation Events: 4 types (clouds, soiling, storms, heatwaves)")
        logger.info("=" * 80)
        
        # ===== STEP 9: Save Processed Data (Optional) =====
        logger.info("\n[STEP 9] Saving processed datasets...")
        
        # Save different feature sets for different models
        forecasting_features.to_csv('data/processed_forecasting.csv', index=False)
        anomaly_features.to_csv('data/processed_anomaly.csv', index=False)
        degradation_features.to_csv('data/processed_degradation.csv', index=False)
        optimization_features.to_csv('data/processed_optimization.csv', index=False)
        
        # Save augmented data
        all_data.to_csv('data/augmented_training_data.csv', index=False)
        
        logger.info("✓ All datasets saved successfully")
        
        # ===== STEP 10: Data Quality Monitoring Dashboard =====
        logger.info("\n[STEP 10] Quality monitoring summary...")
        
        print("\n" + "="*60)
        print("DATA QUALITY DASHBOARD")
        print("="*60)
        print(f"{'Metric':<25} {'Score':<10} {'Status':<15}")
        print("-"*60)
        
        def get_status(score):
            if score >= 0.9: return "✓ Excellent"
            elif score >= 0.7: return "⚠ Good"
            elif score >= 0.5: return "⚠ Fair"
            else: return "✗ Poor"
        
        print(f"{'Completeness':<25} {final_metrics.completeness:<10.3f} {get_status(final_metrics.completeness):<15}")
        print(f"{'Validity':<25} {final_metrics.validity:<10.3f} {get_status(final_metrics.validity):<15}")
        print(f"{'Consistency':<25} {final_metrics.consistency:<10.3f} {get_status(final_metrics.consistency):<15}")
        print(f"{'Timeliness':<25} {final_metrics.timeliness:<10.3f} {get_status(final_metrics.timeliness):<15}")
        print(f"{'Anomaly Score':<25} {final_metrics.anomaly_score:<10.3f} {'(Lower is better)':<15}")
        print("-"*60)
        print(f"{'OVERALL QUALITY':<25} {final_metrics.overall_score:<10.3f} {get_status(final_metrics.overall_score):<15}")
        print("="*60)
        
        logger.info("\n✓ Enhanced Data Pipeline completed successfully!")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        raise
