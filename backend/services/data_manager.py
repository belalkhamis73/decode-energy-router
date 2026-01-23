# backend/services/data_manager.py
""" Streaming Data Manager Service – now with Session Aggregate + Repository """
import logging
import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Deque
from collections import deque

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
import torch
import aiohttp
import redis.asyncio as redis

from .weather_client import WeatherAPIClient
from .subhourly_interpolator import SubHourlyInterpolator
from ..core.user_controls import UserControlSurface
from ..core.fault_scenario import FaultScenario

logger = logging.getLogger("StreamingDataManager")

# ────────────────────────
# DOMAIN AGGREGATE ROOT
# ────────────────────────

@dataclass
class DigitalTwinSession:
    """Aggregate root for a digital twin simulation session."""
    session_id: str
    topology: Dict[str, Any]
    location: Tuple[float, float]
    weather_mode: str

    # Mutable state (owned by aggregate)
    user_controls: Dict[str, Any] = field(default_factory=dict)
    fault_scenario: Dict[str, Any] = field(default_factory=dict)
    asset_states: Dict[str, Any] = field(default_factory=dict)
    model_outputs_history: Deque = field(default_factory=lambda: deque(maxlen=3600))
    constraint_violations_history: List = field(default_factory=list)
    weather_history: Deque = field(default_factory=lambda: deque(maxlen=3600))

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tick_counter: int = 0
    simulation_time: float = 0.0

    # Runtime dependencies (transient, not persisted)
    _weather_stream: Optional["asyncio.Queue"] = None

    def set_weather_stream(self, queue: "asyncio.Queue"):
        self._weather_stream = queue

    def get_weather_stream(self) -> Optional["asyncio.Queue"]:
        return self._weather_stream

    def to_persistent_dict(self) -> Dict[str, Any]:
        """Convert to dict safe for serialization (excludes transient fields)."""
        d = asdict(self)
        d.pop("_weather_stream", None)
        return d

    @classmethod
    def from_persistent_dict(cls, data: Dict[str, Any]) -> "DigitalTwinSession":
        """Reconstruct from persisted dict."""
        # Rehydrate deques
        if "model_outputs_history" in data:
            data["model_outputs_history"] = deque(data["model_outputs_history"], maxlen=3600)
        if "weather_history" in data:
            data["weather_history"] = deque(data["weather_history"], maxlen=3600)
        return cls(**data)


# ────────────────────────
# PERSISTENCE BOUNDARY
# ────────────────────────

class SessionRepository(ABC):
    @abstractmethod
    async def save(self, session: DigitalTwinSession) -> None:
        pass

    @abstractmethod
    async def load(self, session_id: str) -> Optional[DigitalTwinSession]:
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        pass


class FileSessionRepository(SessionRepository):
    def __init__(self, storage_dir: str = "./session_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    async def save(self, session: DigitalTwinSession) -> None:
        path = os.path.join(self.storage_dir, f"{session.session_id}.json")
        data = session.to_persistent_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    async def load(self, session_id: str) -> Optional[DigitalTwinSession]:
        path = os.path.join(self.storage_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return DigitalTwinSession.from_persistent_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def delete(self, session_id: str) -> bool:
        path = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


# ────────────────────────
# STREAMING DATA MANAGER
# ────────────────────────

class StreamingDataManager:
    SUPPORTED_GRIDS = ["ieee14", "ieee30", "ieee118"]

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        api_keys: Optional[Dict[str, str]] = None,
        repository: Optional[SessionRepository] = None,
    ):
        self.redis_url = redis_url
        self.api_keys = api_keys or {}
        self.repository = repository or FileSessionRepository()
        self.redis_client: Optional[redis.Redis] = None
        self.weather_client: Optional[WeatherAPIClient] = None
        self.interpolator = SubHourlyInterpolator()

        # In-memory cache of active sessions (NOT the source of truth)
        self._active_sessions: Dict[str, DigitalTwinSession] = {}
        self._streaming_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Continuing without event streaming.")
            self.redis_client = None

        self.weather_client = WeatherAPIClient(self.api_keys)
        await self.weather_client.__aenter__()

    async def shutdown(self):
        for task in self._streaming_tasks.values():
            task.cancel()
        if self.redis_client:
            await self.redis_client.close()
        if self.weather_client:
            await self.weather_client.__aexit__(None, None, None)

    async def create_session(
        self,
        sid: str,
        topology_name: str,
        location: Tuple[float, float] = (30.0, 31.0),
        weather_mode: str = "live",
    ) -> DigitalTwinSession:
        logger.info(f"✨ Creating Streaming Session {sid}[{topology_name}]")

        topology = self.get_topology(topology_name)
        weather_queue = asyncio.Queue(maxsize=3600)

        session = DigitalTwinSession(
            session_id=sid,
            topology=topology,
            location=location,
            weather_mode=weather_mode,
            user_controls=asdict(UserControlSurface()),
            fault_scenario=asdict(FaultScenario()),
            asset_states={
                "battery_soc": 0.5,
                "battery_temp_c": 25.0,
                "battery_soh": 1.0,
                "battery_cycles": 0.0,
                "diesel_status": "OFF",
                "diesel_warmup_remaining_sec": 0.0,
                "diesel_runtime_sec": 0.0,
                "diesel_fuel_remaining_liters": 1000.0,
                "grid_connected": True,
                "total_solar_generation_kwh": 0.0,
                "total_wind_generation_kwh": 0.0,
                "total_load_served_kwh": 0.0,
            },
        )
        session.set_weather_stream(weather_queue)

        # Save to persistent store
        await self.repository.save(session)
        # Cache in memory
        self._active_sessions[sid] = session

        if weather_mode == "live":
            task = asyncio.create_task(self._stream_live_weather(sid, location, weather_queue))
            self._streaming_tasks[sid] = task
            logger.info(f" > Started live weather stream for {sid}")

        return session

    # ─── SYNC WRAPPERS FOR ROUTES ───────────────────────

    def get_session(self, sid: str) -> Optional[DigitalTwinSession]:
        """Sync wrapper for FastAPI routes."""
        return self._active_sessions.get(sid)

    def list_sessions(self) -> List[str]:
        return list(self._active_sessions.keys())

    def get_session_count(self) -> int:
        return len(self._active_sessions)

    def update_session_state(self, sid: str, updates: Dict[str, Any]) -> bool:
        session = self._active_sessions.get(sid)
        if not session:
            return False
        # Apply updates
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
            elif key in session.asset_states:
                session.asset_states[key] = value
        return True

    def remove_session(self, sid: str) -> bool:
        removed = self._active_sessions.pop(sid, None) is not None
        if removed:
            logger.info(f"🗑️ Removed session {sid}")
        return removed

    # ─── INTERNAL METHODS ────────────────────────────────

    async def _stream_live_weather(self, sid: str, location: Tuple[float, float], queue: asyncio.Queue):
        lat, lon = location
        try:
            while True:
                solar_data = await self.weather_client.fetch_nrel_solar(lat, lon)
                weather_data = await self.weather_client.fetch_openweather(lat, lon)

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

                try:
                    queue.put_nowait(snapshot)
                except asyncio.QueueFull:
                    queue.get_nowait()
                    queue.put_nowait(snapshot)

                if self.redis_client:
                    await self.redis_client.publish(
                        f"weather:{sid}",
                        json.dumps(asdict(snapshot))
                    )

                await asyncio.sleep(300)
        except asyncio.CancelledError:
            logger.info(f"Weather stream for {sid} stopped")
        except Exception as e:
            logger.error(f"Weather stream error: {e}")

    def get_topology(self, grid_name: str) -> Dict[str, Any]:
        grid_name = grid_name.lower().strip()
        if "118" in grid_name:
            net = pn.case118()
            n_buses = 118
        elif "30" in grid_name:
            net = pn.case30()
            n_buses = 30
        else:
            net = pn.case14()
            n_buses = 14

        try:
            mg = pp.topology.create_nxgraph(net, include_lines=True, include_trafo=True, multi=False)
            adj_matrix = nx.to_numpy_array(mg, nodelist=sorted(mg.nodes()))
        except Exception:
            adj_matrix = np.eye(n_buses)

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
            "net": net
        }

    def get_user_control_surface(self, sid: str) -> Optional[UserControlSurface]:
        session = self.get_session(sid)
        if not session:
            return None
        return UserControlSurface(**session.user_controls)

    async def apply_user_override(self, sid: str, parameter: str, value: Any) -> bool:
        session = self.get_session(sid)
        if not session:
            logger.warning(f"Session {sid} not found")
            return False
        controls = session.user_controls

        if parameter not in controls:
            logger.warning(f"Invalid parameter: {parameter}")
            return False

        current_type = type(controls[parameter])
        if not isinstance(value, current_type):
            try:
                value = current_type(value)
            except (ValueError, TypeError):
                logger.warning(f"Type mismatch for {parameter}")
                return False

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

        controls[parameter] = value
        logger.info(f"✅ Applied {parameter}={value} for session {sid}")

        if self.redis_client:
            event = {
                'type': 'user_override',
                'session_id': sid,
                'parameter': parameter,
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            await self.redis_client.publish(f"controls:{sid}", json.dumps(event))
        return True

    async def inject_synthetic_event(self, sid: str, event_type: str, params: Dict[str, Any]) -> bool:
        session = self.get_session(sid)
        if not session:
            return False
        logger.info(f"⚡ Injecting {event_type} event for session {sid}")

        if event_type == "black_swan_weather" and TIMEGAN_AVAILABLE:
            try:
                config = TimeGANConfig(seq_len=24, feature_dim=3)
                generator = TimeGANGenerator(config)
                z = torch.randn(1, 24, 3)
                synthetic = generator(z).detach().numpy()[0]
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
                    session.get_weather_stream().put_nowait(snapshot)
                logger.info(" > Injected 24h synthetic chaos profile")
            except Exception as e:
                logger.error(f"TimeGAN error: {e}")
                return False
        elif event_type in ["line_fault", "generator_trip", "load_spike"]:
            fault = FaultScenario(
                active=True,
                fault_type=event_type,
                magnitude=params.get('magnitude', 0.5),
                affected_buses=params.get('buses', [1]),
                start_time=datetime.now().timestamp(),
                duration_sec=params.get('duration_sec', 60.0)
            )
            session.fault_scenario = asdict(fault)
            if self.redis_client:
                event = {
                    'type': 'synthetic_event',
                    'event_type': event_type,
                    'params': params,
                    'timestamp': datetime.now().isoformat()
                }
                await self.redis_client.publish(f"events:{sid}", json.dumps(event))
            return True
        return True

    def generate_1hz_weather_profile(self, hourly_profile: Dict[str, List[float]], apply_transients: bool = True) -> Dict[str, np.ndarray]:
        result = {}
        for key in ['ghi', 'wind_speed', 'temperature']:
            if key in hourly_profile:
                hourly = np.array(hourly_profile[key])
                hz_1 = self.interpolator.interpolate_to_1hz(hourly)
                if apply_transients:
                    if key == 'ghi':
                        hz_1 = self.interpolator.add_cloud_transients(hz_1)
                    elif key == 'wind_speed':
                        hz_1 = self.interpolator.add_wind_gusts(hz_1)
                result[key] = hz_1
        return result

# Global instance
data_manager = StreamingDataManager()
