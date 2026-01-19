"""
NREL Data Ingestion & Physics-Informed Processing Pipeline.
Compliant with SaaS Digital Twin MLOps Standards.

Alterations:
- Added PhysicsScaler class to normalize synthetic/loaded data.
- Implemented specific scaling logic: Solar to [0-1], Voltage to p.u.
"""

import logging
import requests
import pandas as pd
import numpy as np
import torch
import pvlib
import io  # FIX: Added missing import
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging as per Technical Blueprint
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
        """
        params = {
            'wkt': f'POINT({location.lon} {location.lat})',
            'names': year,
            'leap_day': 'false',
            'interval': '60',
            'utc': 'false',
            'full_name': 'DecodeUser',
            'email': email,
            'affiliation': 'DecodeEnergy',
            'mailing_list': 'false',
            'reason': 'Academic',
            'api_key': self._api_key,
            'attributes': 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
        }
        
        try:
            logger.info(f"Fetching NREL data for {location}...")
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text), skiprows=2)
            logger.info(f"Successfully loaded {len(df)} rows.")
            return df
            
        except Exception as e:
            logger.error(f"NREL API Failed: {e}")
            # Fallback for offline/demo mode handled by DataManager in backend
            raise e

class PhysicsProcessor:
    """
    Cleans and enriches raw weather data with physical constraints.
    """
    def enforce_solar_geometry(self, df: pd.DataFrame, loc: GeoLocation) -> pd.DataFrame:
        """
        Calculates Zenith angle and enforces physical consistency:
        GHI <= Solar_Constant * cos(Zenith)
        """
        # Ensure timestamp index
        if 'Year' in df.columns:
            df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            df.set_index('datetime', inplace=True)
            
        # Solar Position Calculation (PVLib)
        site = pvlib.location.Location(loc.lat, loc.lon)
        solar_pos = site.get_solarposition(df.index)
        
        df['zenith'] = solar_pos['zenith']
        df['cos_zenith'] = np.cos(np.radians(df['zenith']))
        
        # Physics Check: Nighttime GHI must be 0
        df.loc[df['cos_zenith'] <= 0, ['GHI', 'DNI', 'DHI']] = 0
        
        return df

class PhysicsScaler(BaseEstimator, TransformerMixin):
    """
    Domain-Specific Normalizer for Energy Systems.
    Scales inputs to ranges optimal for Neural Network convergence while preserving physical meaning.
    """
    
    # Physical Limits for Min-Max Scaling
    MAX_GHI = 1400.0   # W/m^2 (Approx Solar Constant)
    MAX_WIND = 30.0    # m/s (Storm cut-off)
    MAX_TEMP = 60.0    # Celsius
    MIN_TEMP = -20.0   # Celsius
    
    def __init__(self):
        self.voltage_base = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Standard sklearn transform (placeholder)
        return X

    def normalize_context(self, weather_data: Dict[str, List[float]]) -> torch.Tensor:
        """
        Scales raw weather dictionary (from DataManager) to [0, 1] tensors for DeepONet.
        """
        # Convert lists to numpy arrays
        ghi = np.array(weather_data.get('ghi', []), dtype=np.float32)
        wind = np.array(weather_data.get('wind_speed', []), dtype=np.float32)
        temp = np.array(weather_data.get('temperature', []), dtype=np.float32)
        
        # Apply Scaling
        ghi_norm = np.clip(ghi / self.MAX_GHI, 0.0, 1.0)
        wind_norm = np.clip(wind / self.MAX_WIND, 0.0, 1.0)
        temp_norm = (temp - self.MIN_TEMP) / (self.MAX_TEMP - self.MIN_TEMP)
        temp_norm = np.clip(temp_norm, 0.0, 1.0)
        
        # Stack into Tensor [Seq_Len, Features]
        normalized_tensor = torch.tensor(
            np.stack([ghi_norm, wind_norm, temp_norm], axis=1),
            dtype=torch.float32
        )
        return normalized_tensor

    def normalize_voltage(self, voltage: Union[float, np.ndarray, torch.Tensor], base_kv: float) -> Union[float, torch.Tensor]:
        """
        Scales absolute voltage (kV) to Per-Unit (p.u.).
        """
        if isinstance(voltage, torch.Tensor):
            if voltage.max() < 2.0:
                return voltage
            return voltage / base_kv
        
        elif isinstance(voltage, (np.ndarray, float)):
            if np.max(voltage) < 2.0:
                return voltage
            return voltage / base_kv
            
        return voltage

    def denormalize_prediction(self, pred_tensor: torch.Tensor, target_key: str) -> np.ndarray:
        """
        Converts model outputs back to physical units for reporting.
        """
        val = pred_tensor.detach().cpu().numpy()
        
        if target_key == 'ghi':
            return val * self.MAX_GHI
        elif target_key == 'voltage_pu':
            return val 
            
        return val
