"""
Data Manager Service.
Central repository for 'Valuable Inputs' (Grid Topologies, Weather Contexts, & Asset Profiles).
Acts as the STATEFUL SESSION STORE for the Digital Twin.

Key Integrations:
1. Session Management: Stores persistent state (Battery SOC, Diesel Status) per user.
2. Grid Topology: Loads IEEE 14/118 via Pandapower.
3. Weather Generation: Uses TimeGAN for 'Black Swan' chaos scenarios.
4. V2G Modeling: Generates stochastic EV availability profiles.
"""

import logging
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
import torch
from typing import Dict, Any, List, Optional

# --- Import TimeGAN for Chaos Generation ---
try:
    from ml_models.training.timegan_generator import TimeGANGenerator, TimeGANConfig
    TIMEGAN_AVAILABLE = True
except ImportError:
    TIMEGAN_AVAILABLE = False

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataManager")

class DataManager:
    """
    The 'Context Engine' and 'Session Store' for the SaaS.
    Decouples the Model from the Data Source and maintains Simulation State.
    """
    
    SUPPORTED_GRIDS = ["ieee14", "ieee30", "ieee118"]

    def __init__(self):
        # In-Memory Database for active sessions
        # Structure: { session_id: { topology, weather, battery_soc, diesel_status ... } }
        self._sessions: Dict[str, Any] = {} 

    def create_session(self, sid: str, topology_name: str, weather_profile: str) -> Dict[str, Any]:
        """
        Initializes a persistent session state for a Digital Twin.
        Called by main.py during /configure.
        """
        logger.info(f"✨ Creating Session {sid} [{topology_name}]")
        
        # 1. Load Topology
        topo = self.get_topology(topology_name)
        
        # 2. Generate Weather Context
        weather = self.get_weather_profile(weather_profile)
        
        # 3. Initialize Stateful Assets (The "Memory" of the simulation)
        session_state = {
            "topology": topo,
            "weather": weather,
            # Stateful Assets
            "battery_soc": 0.5,       # Starts at 50%
            "battery_temp_c": 25.0,   # Starts at 25C
            "battery_soh": 1.0,       # Starts at 100% Health
            "diesel_status": "OFF",   # Starts OFF
            "tick_counter": 0
        }
        
        # 4. Save to Memory
        self._sessions[sid] = session_state
        return session_state

    def get_session(self, sid: str) -> Optional[Dict[str, Any]]:
        """Retrieves the full state context for a specific session."""
        return self._sessions.get(sid)

    def get_topology(self, grid_name: str) -> Dict[str, Any]:
        """
        Loads a standard IEEE grid and converts it to a graph representation.
        """
        grid_name = grid_name.lower().strip()
        
        # 1. Load Standard Network from Pandapower
        if "118" in grid_name:
            net = pn.case118()
            n_buses = 118
        elif "30" in grid_name:
            net = pn.case30()
            n_buses = 30
        else:
            net = pn.case14()
            n_buses = 14
            
        # 2. Construct Graph Representation (Physics Core)
        # Create NetworkX graph from lines for adjacency matrix
        try:
            mg = pp.topology.create_nxgraph(net, include_lines=True, include_trafo=True, multi=False)
            adj_matrix = nx.to_numpy_array(mg, nodelist=sorted(mg.nodes()))
        except Exception:
            # Fallback if networkx/pandapower version mismatch
            adj_matrix = np.eye(n_buses)
        
        return {
            "name": grid_name,
            "n_buses": int(n_buses),
            "adj_matrix": adj_matrix
        }

    def get_weather_profile(self, profile_type: str = "solar_egypt", scenario: str = "normal") -> Dict[str, List[float]]:
        """
        Generates environmental context.
        """
        # A. BLACK SWAN SCENARIO (TimeGAN)
        if scenario == "black_swan" and TIMEGAN_AVAILABLE:
            try:
                config = TimeGANConfig(seq_len=24, feature_dim=3)
                generator = TimeGANGenerator(config)
                z = torch.randn(1, 24, 3) 
                synthetic_data = generator(z).detach().numpy()[0]
                
                ghi = np.clip(synthetic_data[:, 0] * 1400, 0, 1400)
                wind = np.clip(synthetic_data[:, 1] * 30, 0, 30)
                temp = (synthetic_data[:, 2] * 40) + 10
                logger.info("   > Generated Synthetic Chaos Event via TimeGAN")
            except Exception:
                ghi, wind, temp = self._generate_standard_profile(profile_type)
        else:
            # B. STANDARD SCENARIO
            ghi, wind, temp = self._generate_standard_profile(profile_type)

        # C. V2G MODELING
        v2g_profile = self._generate_v2g_availability()

        return {
            "time_step": np.linspace(0, 23, 24).tolist(),
            "ghi": ghi.tolist(),
            "wind_speed": wind.tolist(),
            "temperature": temp.tolist(),
            "v2g_availability_kw": v2g_profile.tolist()
        }

    def _generate_standard_profile(self, profile_type: str):
        """Helper to generate standard Gaussian/Sinusoidal curves."""
        t = np.linspace(0, 23, 24)
        if "solar" in profile_type:
            ghi = 1050 * np.exp(-0.5 * ((t - 13) / 3.5)**2)
            temp = 25 + (ghi / 50) + np.random.normal(0, 1, 24)
            wind = 5 + 2 * np.sin(t / 4) + np.random.normal(0, 1, 24)
        elif "wind" in profile_type:
            ghi = 600 * np.exp(-0.5 * ((t - 12) / 4)**2)
            wind = 12 + 8 * np.sin(t / 3) + np.random.normal(0, 3, 24)
            temp = 15 + np.random.normal(0, 2, 24)
        else:
            ghi = np.zeros(24); wind = np.zeros(24); temp = np.ones(24) * 20
            
        return np.clip(ghi, 0, 1400), np.clip(wind, 0, 50), temp

    def _generate_v2g_availability(self) -> np.ndarray:
        """Simulates availability of EV Fleet for Vehicle-to-Grid services."""
        t = np.linspace(0, 23, 24)
        work_hours = 1 / (1 + np.exp(-(t - 8))) - 1 / (1 + np.exp(-(t - 18)))
        base_capacity = 140.0 
        availability = base_capacity * (0.3 + 0.7 * work_hours) 
        noise = np.random.normal(0, 10, 24)
        return np.clip(availability + noise, 0, base_capacity)

# Global Instance
data_manager = DataManager()
