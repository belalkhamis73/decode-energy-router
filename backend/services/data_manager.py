"""
Streaming Data Manager Service
Real-time data ingestion, state management, and event streaming for Digital Twin SaaS.

Key Features:
1. Real-time weather APIs (NREL NSRDB, OpenWeather) with sub-hourly interpolation
2. Stochastic noise injection (cloud transients, wind gusts, load variations)
3. Redis event streaming for pub/sub architecture
4. Historical buffer management (1Hz data, 1-hour rolling window)
5. Comprehensive user control surface for all simulation parameters
6. Synthetic event injection (black swan scenarios, fault conditions)
7. Session-based state persistence with full parameter tracking
"""

import logging
import asyncio
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
import torch
import aiohttp
import redis.asyncio as redis
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from scipy.interpolate import interp1d

# --- Import TimeGAN for Chaos Generation ---
try:
    from ml_models.training.timegan_generator import TimeGANGenerator, TimeGANConfig
    TIMEGAN_AVAILABLE = True
except ImportError:
    TIMEGAN_AVAILABLE = False
    logging.warning("TimeGAN not available - synthetic chaos events disabled")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StreamingDataManager")


@dataclass
class WeatherSnapshot:
    """Real-time weather observation"""
    timestamp: float
    ghi: float  # W/m²
    dni: float  # W/m²
    dhi: float  # W/m²
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    temperature: float  # °C
    humidity: float  # %
    pressure: float  # hPa
    cloud_cover: float  # 0-1


@dataclass
class UserControlSurface:
    """All user-controllable simulation parameters"""
    # Environmental modifiers
    cloud_cover_factor: float = 1.0  # 0.0-2.0
    wind_speed_multiplier: float = 1.0  # 0.0-2.0
    temperature_offset: float = 0.0  # -10 to +10 °C
    load_multiplier: float = 1.0  # 0.5-2.0
    
    # Battery controls
    battery_cooling_override: float = 1.0  # 0.0-2.0 (cooling effectiveness)
    min_soc_reserve: float = 0.2  # 0.0-0.5
    max_soc_limit: float = 0.95  # 0.5-1.0
    max_charge_rate_kw: float = 50.0  # 0-100 kW
    max_discharge_rate_kw: float = 50.0  # 0-100 kW
    
    # Diesel controls
    diesel_auto_start_enabled: bool = True
    diesel_min_runtime_sec: float = 300.0  # 5 min minimum
    diesel_warmup_sec: float = 30.0
    diesel_cooldown_sec: float = 60.0
    
    # V2G controls
    v2g_participation_rate: float = 0.8  # 0.0-1.0
    v2g_min_departure_soc: float = 0.8  # Reserve for departing EVs
    
    # Grid controls
    grid_import_limit_kw: float = 100.0
    grid_export_limit_kw: float = 50.0


@dataclass
class FaultScenario:
    """Active fault/disturbance scenario"""
    active: bool = False
    fault_type: Optional[str] = None  # 'line_fault', 'generator_trip', 'load_spike'
    magnitude: float = 0.0  # Severity 0.0-1.0
    affected_buses: List[int] = None
    start_time: Optional[float] = None
    duration_sec: float = 0.0
    
    def __post_init__(self):
        if self.affected_buses is None:
            self.affected_buses = []


class WeatherAPIClient:
    """Async client for real-time weather data"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.nrel_api_key = api_keys.get('nrel', '')
        self.openweather_api_key = api_keys.get('openweather', '')
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_nrel_solar(self, lat: float, lon: float) -> Dict[str, float]:
        """Fetch solar irradiance from NREL NSRDB API"""
        if not self.nrel_api_key or not self.session:
            return self._fallback_solar()
            
        try:
            url = "https://developer.nrel.gov/api/solar/solar_resource/v1.json"
            params = {
                'api_key': self.nrel_api_key,
                'lat': lat,
                'lon': lon
            }
            
            async with self.session.get(url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Parse NREL response
                    outputs = data.get('outputs', {})
                    return {
                        'ghi': outputs.get('avg_ghi', {}).get('annual', 0) / 365 * 1000,
                        'dni': outputs.get('avg_dni', {}).get('annual', 0) / 365 * 1000,
                        'dhi': outputs.get('avg_ghi', {}).get('annual', 0) / 365 * 300,
                    }
        except Exception as e:
            logger.warning(f"NREL API error: {e}")
            
        return self._fallback_solar()
    
    async def fetch_openweather(self, lat: float, lon: float) -> Dict[str, float]:
        """Fetch current weather from OpenWeather API"""
        if not self.openweather_api_key or not self.session:
            return self._fallback_weather()
            
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'appid': self.openweather_api_key,
                'lat': lat,
                'lon': lon,
                'units': 'metric'
            }
            
            async with self.session.get(url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'pressure': data['main']['pressure'],
                        'wind_speed': data['wind']['speed'],
                        'wind_direction': data['wind'].get('deg', 0),
                        'cloud_cover': data['clouds']['all'] / 100.0
                    }
        except Exception as e:
            logger.warning(f"OpenWeather API error: {e}")
            
        return self._fallback_weather()
    
    @staticmethod
    def _fallback_solar() -> Dict[str, float]:
        """Synthetic solar data when API unavailable"""
        hour = datetime.now().hour
        ghi = max(0, 1000 * np.sin(np.pi * (hour - 6) / 12))
        return {'ghi': ghi, 'dni': ghi * 0.8, 'dhi': ghi * 0.2}
    
    @staticmethod
    def _fallback_weather() -> Dict[str, float]:
        """Synthetic weather when API unavailable"""
        return {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.0,
            'wind_speed': 5.0,
            'wind_direction': 180.0,
            'cloud_cover': 0.3
        }


class SubHourlyInterpolator:
    """Generates 1Hz data from hourly profiles with stochastic realism"""
    
    @staticmethod
    def interpolate_to_1hz(hourly_data: np.ndarray, 
                          noise_level: float = 0.05) -> np.ndarray:
        """
        Convert hourly data to 1Hz with cubic interpolation + noise
        
        Args:
            hourly_data: Array of hourly values (length 24)
            noise_level: Gaussian noise std as fraction of signal
            
        Returns:
            1Hz array for 24 hours (86400 samples)
        """
        hours = len(hourly_data)
        t_hourly = np.arange(hours)
        t_1hz = np.linspace(0, hours - 1, hours * 3600)
        
        # Cubic interpolation
        interpolator = interp1d(t_hourly, hourly_data, kind='cubic', 
                               fill_value='extrapolate')
        smooth_1hz = interpolator(t_1hz)
        
        # Add realistic stochastic variations
        noise = np.random.normal(0, noise_level * np.mean(np.abs(hourly_data)), 
                                len(smooth_1hz))
        
        return np.clip(smooth_1hz + noise, 0, None)
    
    @staticmethod
    def add_cloud_transients(ghi_1hz: np.ndarray, 
                            event_rate: float = 0.01) -> np.ndarray:
        """
        Inject realistic cloud passing events
        
        Args:
            ghi_1hz: 1Hz GHI array
            event_rate: Probability of cloud event per second
            
        Returns:
            Modified GHI with cloud transients
        """
        result = ghi_1hz.copy()
        n_samples = len(ghi_1hz)
        
        # Generate cloud events
        events = np.random.random(n_samples) < event_rate
        event_indices = np.where(events)[0]
        
        for idx in event_indices:
            # Cloud shadow: 30-70% reduction for 10-60 seconds
            duration = int(np.random.uniform(10, 60))
            reduction = np.random.uniform(0.3, 0.7)
            
            end_idx = min(idx + duration, n_samples)
            # Smooth transition
            ramp = np.linspace(1, reduction, duration // 2)
            ramp = np.concatenate([ramp, ramp[::-1]])
            
            actual_duration = end_idx - idx
            if actual_duration > 0:
                multiplier = np.interp(np.arange(actual_duration), 
                                      np.arange(len(ramp)), ramp)
                result[idx:end_idx] *= multiplier[:actual_duration]
        
        return result
    
    @staticmethod
    def add_wind_gusts(wind_1hz: np.ndarray, 
                      gust_rate: float = 0.005) -> np.ndarray:
        """Inject realistic wind gust events"""
        result = wind_1hz.copy()
        n_samples = len(wind_1hz)
        
        events = np.random.random(n_samples) < gust_rate
        event_indices = np.where(events)[0]
        
        for idx in event_indices:
            # Gust: 1.3-2.0x increase for 5-20 seconds
            duration = int(np.random.uniform(5, 20))
            amplification = np.random.uniform(1.3, 2.0)
            
            end_idx = min(idx + duration, n_samples)
            # Smooth ramp up/down
            ramp = np.concatenate([
                np.linspace(1, amplification, duration // 3),
                np.ones(duration // 3) * amplification,
                np.linspace(amplification, 1, duration - 2 * (duration // 3))
            ])
            
            actual_duration = end_idx - idx
            if actual_duration > 0:
                multiplier = np.interp(np.arange(actual_duration),
                                      np.arange(len(ramp)), ramp)
                result[idx:end_idx] *= multiplier[:actual_duration]
        
        return np.clip(result, 0, 50)


class StreamingDataManager:
    """
    Enhanced Data Manager with real-time streaming capabilities.
    The 'Live Context Engine' and 'Session Store' for the SaaS.
    """
    
    SUPPORTED_GRIDS = ["ieee14", "ieee30", "ieee118"]
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 api_keys: Optional[Dict[str, str]] = None):
        # Session storage
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
        # Redis for event streaming
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Weather API client
        self.api_keys = api_keys or {}
        self.weather_client: Optional[WeatherAPIClient] = None
        
        # Interpolator
        self.interpolator = SubHourlyInterpolator()
        
        # Background tasks
        self._streaming_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize(self):
        """Async initialization"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Continuing without event streaming.")
            self.redis_client = None
            
        self.weather_client = WeatherAPIClient(self.api_keys)
        await self.weather_client.__aenter__()
        
    async def shutdown(self):
        """Cleanup resources"""
        # Stop all streaming tasks
        for task in self._streaming_tasks.values():
            task.cancel()
            
        if self.redis_client:
            await self.redis_client.close()
            
        if self.weather_client:
            await self.weather_client.__aexit__(None, None, None)
    
    async def create_session(self, 
                            sid: str, 
                            topology_name: str,
                            location: Tuple[float, float] = (30.0, 31.0),  # Cairo
                            weather_mode: str = "live") -> Dict[str, Any]:
        """
        Initialize a persistent session with full control surface.
        
        Args:
            sid: Session identifier
            topology_name: Grid topology (ieee14/30/118)
            location: (latitude, longitude) for weather APIs
            weather_mode: 'live' (API), 'synthetic', or 'black_swan'
        """
        logger.info(f"✨ Creating Streaming Session {sid} [{topology_name}]")
        
        # 1. Load topology
        topology = self.get_topology(topology_name)
        
        # 2. Initialize weather stream
        weather_queue = asyncio.Queue(maxsize=3600)  # 1 hour buffer
        
        # 3. Initialize comprehensive session state
        session_state = {
            'topology': topology,
            'location': location,
            'weather_mode': weather_mode,
            'weather_stream': weather_queue,
            
            # User control surface (all tunable parameters)
            'user_controls': asdict(UserControlSurface()),
            
            # Fault scenario
            'fault_scenario': asdict(FaultScenario()),
            
            # Asset states (persistent simulation state)
            'asset_states': {
                'battery_soc': 0.5,
                'battery_temp_c': 25.0,
                'battery_soh': 1.0,
                'battery_cycles': 0.0,
                'diesel_status': 'OFF',
                'diesel_warmup_remaining_sec': 0.0,
                'diesel_runtime_sec': 0.0,
                'diesel_fuel_remaining_liters': 1000.0,
                'grid_connected': True,
                'total_solar_generation_kwh': 0.0,
                'total_wind_generation_kwh': 0.0,
                'total_load_served_kwh': 0.0,
            },
            
            # Historical buffers
            'model_outputs_history': deque(maxlen=3600),  # 1 hour at 1Hz
            'constraint_violations_history': [],
            'weather_history': deque(maxlen=3600),
            
            # Metadata
            'created_at': datetime.now().isoformat(),
            'tick_counter': 0,
            'simulation_time': 0.0,
        }
        
        self._sessions[sid] = session_state
        
        # 4. Start background weather streaming
        if weather_mode == "live":
            task = asyncio.create_task(
                self._stream_live_weather(sid, location, weather_queue)
            )
            self._streaming_tasks[sid] = task
            logger.info(f"   > Started live weather stream for {sid}")
        
        return session_state
    
    async def _stream_live_weather(self, 
                                   sid: str,
                                   location: Tuple[float, float],
                                   queue: asyncio.Queue):
        """Background task: continuously fetch and interpolate weather data"""
        lat, lon = location
        
        try:
            while True:
                # Fetch current conditions every 5 minutes
                solar_data = await self.weather_client.fetch_nrel_solar(lat, lon)
                weather_data = await self.weather_client.fetch_openweather(lat, lon)
                
                # Create snapshot
                snapshot = WeatherSnapshot(
                    timestamp=datetime.now().timestamp(),
                    ghi=solar_data['ghi'],
                    dni=solar_data['dni'],
                    dhi=solar_data['dhi'],
                    wind_speed=weather_data['wind_speed'],
                    wind_direction=weather_data['wind_direction'],
                    temperature=weather_data['temperature'],
                    humidity=weather_data['humidity'],
                    pressure=weather_data['pressure'],
                    cloud_cover=weather_data['cloud_cover']
                )
                
                # Add to queue (non-blocking)
                try:
                    queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    # Remove oldest if full
                    queue.get_nowait()
                    queue.put_nowait(snapshot)
                
                # Publish to Redis
                if self.redis_client:
                    await self.redis_client.publish(
                        f"weather:{sid}",
                        json.dumps(asdict(snapshot))
                    )
                
                # Wait 5 minutes
                await asyncio.sleep(300)
                
        except asyncio.CancelledError:
            logger.info(f"Weather stream for {sid} stopped")
        except Exception as e:
            logger.error(f"Weather stream error: {e}")
    
    def get_session(self, sid: str) -> Optional[Dict[str, Any]]:
        """Retrieve full session state"""
        return self._sessions.get(sid)
    
    def get_topology(self, grid_name: str) -> Dict[str, Any]:
        """Load IEEE grid topology"""
        grid_name = grid_name.lower().strip()
        
        # Load from pandapower
        if "118" in grid_name:
            net = pn.case118()
            n_buses = 118
        elif "30" in grid_name:
            net = pn.case30()
            n_buses = 30
        else:
            net = pn.case14()
            n_buses = 14
        
        # Create graph representation
        try:
            mg = pp.topology.create_nxgraph(net, include_lines=True, 
                                           include_trafo=True, multi=False)
            adj_matrix = nx.to_numpy_array(mg, nodelist=sorted(mg.nodes()))
        except Exception:
            adj_matrix = np.eye(n_buses)
        
        # Extract line parameters for physics model
        lines_data = []
        for idx, line in net.line.iterrows():
            lines_data.append({
                'from_bus': int(line.from_bus),
                'to_bus': int(line.to_bus),
                'r_ohm': float(line.r_ohm_per_km * line.length_km),
                'x_ohm': float(line.x_ohm_per_km * line.length_km),
                'max_i_ka': float(line.max_i_ka)
            })
        
        return {
            "name": grid_name,
            "n_buses": int(n_buses),
            "adj_matrix": adj_matrix,
            "lines": lines_data,
            "net": net  # Keep pandapower net for power flow
        }
    
    async def get_live_weather_snapshot(self, sid: str) -> Optional[WeatherSnapshot]:
        """Get most recent weather observation for session"""
        session = self.get_session(sid)
        if not session:
            return None
            
        queue = session['weather_stream']
        
        # Non-blocking peek at most recent
        if queue.qsize() > 0:
            # Get all items and put back except last
            items = []
            while not queue.empty():
                items.append(queue.get_nowait())
            
            # Put back
            for item in items:
                queue.put_nowait(item)
                
            return items[-1]
        
        # Fallback: generate synthetic
        return self._generate_synthetic_snapshot()
    
    def _generate_synthetic_snapshot(self) -> WeatherSnapshot:
        """Fallback synthetic weather"""
        hour = datetime.now().hour
        ghi = max(0, 1000 * np.sin(np.pi * (hour - 6) / 12))
        
        return WeatherSnapshot(
            timestamp=datetime.now().timestamp(),
            ghi=ghi,
            dni=ghi * 0.8,
            dhi=ghi * 0.2,
            wind_speed=5.0 + 3.0 * np.sin(hour / 4),
            wind_direction=180.0,
            temperature=25.0 + 5.0 * np.sin(np.pi * (hour - 6) / 12),
            humidity=60.0,
            pressure=1013.0,
            cloud_cover=0.3
        )
    
    def get_user_control_surface(self, sid: str) -> Optional[UserControlSurface]:
        """Get all user-controllable parameters"""
        session = self.get_session(sid)
        if not session:
            return None
            
        return UserControlSurface(**session['user_controls'])
    
    async def apply_user_override(self, 
                                  sid: str,
                                  parameter: str,
                                  value: Any) -> bool:
        """
        Validate and apply user parameter change.
        Publishes change event to Redis.
        """
        session = self.get_session(sid)
        if not session:
            logger.warning(f"Session {sid} not found")
            return False
        
        controls = session['user_controls']
        
        # Validation
        if parameter not in controls:
            logger.warning(f"Invalid parameter: {parameter}")
            return False
        
        # Type-specific validation
        current_type = type(controls[parameter])
        if not isinstance(value, current_type):
            try:
                value = current_type(value)
            except (ValueError, TypeError):
                logger.warning(f"Type mismatch for {parameter}")
                return False
        
        # Range validation
        validation_rules = {
            'cloud_cover_factor': (0.0, 2.0),
            'wind_speed_multiplier': (0.0, 2.0),
            'temperature_offset': (-10.0, 10.0),
            'load_multiplier': (0.5, 2.0),
            'battery_cooling_override': (0.0, 2.0),
            'min_soc_reserve': (0.0, 0.5),
            'max_soc_limit': (0.5, 1.0),
            'max_charge_rate_kw': (0.0, 100.0),
            'max_discharge_rate_kw': (0.0, 100.0),
            'v2g_participation_rate': (0.0, 1.0),
            'v2g_min_departure_soc': (0.5, 1.0),
        }
        
        if parameter in validation_rules:
            min_val, max_val = validation_rules[parameter]
            if not (min_val <= value <= max_val):
                logger.warning(f"{parameter} out of range [{min_val}, {max_val}]")
                return False
        
        # Apply change
        controls[parameter] = value
        logger.info(f"✅ Applied {parameter} = {value} for session {sid}")
        
        # Publish event
        if self.redis_client:
            event = {
                'type': 'user_override',
                'session_id': sid,
                'parameter': parameter,
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            await self.redis_client.publish(
                f"controls:{sid}",
                json.dumps(event)
            )
        
        return True
    
    async def inject_synthetic_event(self,
                                     sid: str,
                                     event_type: str,
                                     params: Dict[str, Any]) -> bool:
        """
        Trigger synthetic disturbance or black swan event.
        
        Event types:
        - 'black_swan_weather': TimeGAN chaos injection
        - 'line_fault': Transmission line failure
        - 'generator_trip': Generator sudden disconnect
        - 'load_spike': Sudden load increase
        - 'frequency_disturbance': Grid frequency deviation
        """
        session = self.get_session(sid)
        if not session:
            return False
        
        logger.info(f"🌩️  Injecting {event_type} event for session {sid}")
        
        if event_type == "black_swan_weather" and TIMEGAN_AVAILABLE:
            # Generate chaotic weather profile
            try:
                config = TimeGANConfig(seq_len=24, feature_dim=3)
                generator = TimeGANGenerator(config)
                z = torch.randn(1, 24, 3)
                synthetic = generator(z).detach().numpy()[0]
                
                # Create extreme weather snapshots
                for i in range(24):
                    snapshot = WeatherSnapshot(
                        timestamp=datetime.now().timestamp() + i * 3600,
                        ghi=max(0, synthetic[i, 0] * 1400),
                        dni=max(0, synthetic[i, 0] * 1200),
                        dhi=max(0, synthetic[i, 0] * 200),
                        wind_speed=max(0, synthetic[i, 1] * 30),
                        wind_direction=180.0,
                        temperature=10 + synthetic[i, 2] * 40,
                        humidity=60.0,
                        pressure=1013.0,
                        cloud_cover=max(0, min(1, synthetic[i, 0]))
                    )
                    
                    session['weather_stream'].put_nowait(snapshot)
                
                logger.info("   > Injected 24h synthetic chaos profile")
                
            except Exception as e:
                logger.error(f"TimeGAN error: {e}")
                return False
        
        elif event_type in ["line_fault", "generator_trip", "load_spike"]:
            # Configure fault scenario
            fault = FaultScenario(
                active=True,
                fault_type=event_type,
                magnitude=params.get('magnitude', 0.5),
                affected_buses=params.get('buses', [1]),
                start_time=datetime.now().timestamp(),
                duration_sec=params.get('duration_sec', 60.0)
            )
            session['fault_scenario'] = asdict(fault)
            
        # Publish event
        if self.redis_client:
            event = {
                'type': 'synthetic_event',
                'event_type': event_type,
                'params': params,
                'timestamp': datetime.now().isoformat()
            }
            await self.redis_client.publish(
                f"events:{sid}",
                json.dumps(event)
            )
        
        return True
    
    def generate_1hz_weather_profile(self, 
                                     hourly_profile: Dict[str, List[float]],
                                     apply_transients: bool = True) -> Dict[str, np.ndarray]:
        """
        Convert hourly weather to 1Hz with realistic stochastic variations.
        
        Returns:
            Dictionary with 1Hz arrays for 24 hours (86,400 samples each)
        """
        result = {}
        
        for key in ['ghi', 'wind_speed', 'temperature']:
            if key in hourly_profile:
                hourly = np.array(hourly_profile[key])
                
                # Base interpolation
                hz_1 = self.interpolator.interpolate_to_1hz(hourly)
                
                # Add transients
                if apply_transients:
                    if key == 'ghi':
                        hz_1 = self.interpolator.add_cloud_transients(hz_1)
                    elif key == 'wind_speed':
                        hz_1 = self.interpolator.add_wind_gusts(hz_1)
                
                result[key] = hz_1
        
        return result


# Global instance
data_manager = StreamingDataManager()

