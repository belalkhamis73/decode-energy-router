"""
NREL Data Ingestion & Physics-Informed Processing Pipeline.
Compliant with SaaS Digital Twin MLOps Standards.
"""

import logging
import requests
import pandas as pd
import numpy as np
import torch
import pvlib
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin

# [span_3](start_span)Configure logging as per Technical Blueprint[span_3](end_span)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants (Should ideally be imported from physics-core)
SOLAR_CONSTANT_GHI = 1361.0  # W/m^2

@dataclass(frozen=True)
class GeoLocation:
    """Immutable value object for site coordinates."""
    lat: float
    lon: float

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
            response.raise_for_status() # Fail Fast on 4xx/5xx errors
            
            # Save raw bytes to avoid encoding issues, then read
            filename = f"nrel_raw_{year}.csv"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            # Skip metadata rows specific to NREL format
            return pd.read_csv(filename, skiprows=2)

        except requests.exceptions.RequestException as e:
            logger.error(f"NREL API Request failed: {e}")
            raise

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
        if 'time_index' not in df.columns:
            # Construct index from NREL columns
            df['time_index'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        
        df = df.set_index('time_index') if 'time_index' in df.columns else df
        
        # Calculate ground truth solar position
        site = pvlib.location.Location(location.lat, location.lon, tz='UTC')
        solar_position = site.get_solarposition(df.index)
        
        # Constraint: Elevation <= 0 implies Night
        is_night = solar_position['apparent_elevation'] <= 0
        
        cols_to_fix = ['GHI', 'DNI', 'DHI']
        for col in cols_to_fix:
            if col in df.columns:
                # Remove sensor noise (negative values)
                df.loc[df[col] < 0, col] = 0.0
                # Apply hard physical constraint
                df.loc[is_night, col] = 0.0
        
        # Feature Engineering: Cosine of Zenith Angle
        df['cos_zenith'] = np.cos(np.radians(solar_position['zenith'])).clip(lower=0.0)
        
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
        self.temp_mean = df['Temperature'].mean()
        self.temp_std = df['Temperature'].std()
        self.wind_max = df['Wind Speed'].max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()
        
        # Scale Irradiance by Solar Constant (Preserves GHI = DNI*cos(theta) + DHI structure)
        for col in ['GHI', 'DNI', 'DHI']:
            if col in df_scaled.columns:
                df_scaled[col] = df_scaled[col] / SOLAR_CONSTANT_GHI
        
        # Standard Scaling for Temperature
        if 'Temperature' in df_scaled.columns:
            df_scaled['Temperature'] = (df_scaled['Temperature'] - self.temp_mean) / self.temp_std
            
        # MinMax Scaling for Wind (Bounded [0, 1])
        if 'Wind Speed' in df_scaled.columns:
            df_scaled['Wind Speed'] = df_scaled['Wind Speed'] / (self.wind_max + 1e-6)
            
        return df_scaled

class SolarDataset(Dataset):
    """
    PyTorch Dataset for LSTM-PINN training.
    Principle: Interface Segregation (Complies with torch.utils.data.Dataset).
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 24, target_col: str = 'GHI'):
        self.seq_len = seq_len
        self.feature_cols = ['DNI', 'DHI', 'Temperature', 'Wind Speed', 'cos_zenith']
        
        # Validation
        if not all(col in df.columns for col in self.feature_cols):
            raise ValueError(f"DataFrame missing required columns: {self.feature_cols}")

        self.features = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df[[target_col]].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: History window [seq_len, features]
        x = self.features[idx : idx + self.seq_len]
        
        # y: Forecast target at t+1
        y = self.target[idx + self.seq_len]
        
        # physics_input: Inputs needed for PDE loss calculation at t+1 (e.g., cos_zenith)
        # Assuming cos_zenith is the last column (-1)
        physics_input = self.features[idx + self.seq_len, -1]
        
        return x, y, physics_input

# --- Usage Example (to be triggered by main orchestration script) ---
if __name__ == "__main__":
    # Configuration (In production, load from environment variables)
    LOC = GeoLocation(lat=31.2357, lon=30.0444)
    API_KEY = "DEMO_KEY" # Replace with valid key
    
    try:
        # Dependency Injection / Composition
        client = NRELClient(API_KEY)
        cleaner = PhysicsProcessor()
        scaler = PhysicsScaler()
        
        # Execution Pipeline
        raw_df = client.fetch_data(LOC, "2022", "user@example.com")
        clean_df = cleaner.enforce_solar_geometry(raw_df, LOC)
        norm_df = scaler.fit(clean_df).transform(clean_df)
        
        dataset = SolarDataset(norm_df)
        logger.info(f"Dataset created with {len(dataset)} sequences.")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
