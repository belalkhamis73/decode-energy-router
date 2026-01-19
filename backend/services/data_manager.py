"""
Data Manager Service.
Central repository for 'Valuable Inputs' (Grid Topologies, Weather Contexts, & Asset Profiles).

Key Integrations:
1. Grid Topology: Loads IEEE 14/118 via Pandapower.
2. Weather Generation: Uses TimeGAN for 'Black Swan' chaos scenarios.
3. V2G Modeling: Generates stochastic EV availability profiles based on office/home patterns.
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
# Wrapped in try/except to prevent system crash if ML dependencies are missing
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
    The 'Context Engine' for the SaaS.
    Decouples the Model from the Data Source, allowing users to 'Bring Your Own Context'
    or select from standard benchmarks.
    """
    
    SUPPORTED_GRIDS = ["ieee14", "ieee30", "ieee118"]

    def __init__(self):
        self._cache = {}

    def get_topology(self, grid_name: str) -> Dict[str, Any]:
        """
        Loads a standard IEEE grid and converts it to a graph representation.
        Correctly distinguishes between IEEE 14 and 118 bus systems.
        """
        grid_name = grid_name.lower().strip()
        logger.info(f"🔌 Loading Grid Topology: {grid_name.upper()}...")

        # 1. Load Standard Network from Pandapower
        if "118" in grid_name:
            net = pn.case118()
            n_buses = 118
        elif "30" in grid_name:
            net = pn.case30()
            n_buses = 30
        else:
            # Default to IEEE 14 (Standard Microgrid)
            net = pn.case14()
            n_buses = 14
            
        # 2. Construct Graph Representation (Physics Core)
        # Create NetworkX graph from lines for adjacency matrix
        mg = pp.topology.create_nxgraph(net, include_lines=True, include_trafo=True, multi=False)
        adj_matrix = nx.to_numpy_array(mg, nodelist=sorted(mg.nodes()))
        
        # 3. Extract Physics Parameters
        n_lines = len(net.line)
        r_lines = net.line['r_ohm_per_km'].values if 'r_ohm_per_km' in net.line else np.zeros(n_lines)
        x_lines = net.line['x_ohm_per_km'].values if 'x_ohm_per_km' in net.line else np.zeros(n_lines)
        
        topology_data = {
            "name": grid_name,
            "n_buses": int(n_buses),
            "n_lines": int(n_lines),
            "adj_matrix": adj_matrix,
            "physics_params": {
                "r_lines": r_lines,
                "x_lines": x_lines
            }
        }
        
        logger.info(f"   > Loaded {n_buses} Buses and {n_lines} Lines.")
        return topology_data

    def get_weather_profile(self, profile_type: str = "solar_egypt", scenario: str = "normal") -> Dict[str, List[float]]:
        """
        Generates environmental context.
        
        Args:
            profile_type: 'solar_egypt', 'wind_north'
            scenario: 'normal' (NREL-like curves) or 'black_swan' (TimeGAN Chaos)
            
        Returns:
            Dictionary with 24-step time-series for GHI, Wind, Temp, and V2G Availability.
        """
        logger.info(f"☀️  Generating Context: {profile_type} [{scenario.upper()}]")
        
        # A. BLACK SWAN SCENARIO (TimeGAN)
        if scenario == "black_swan" and TIMEGAN_AVAILABLE:
            try:
                # Initialize Generator with default config
                config = TimeGANConfig(seq_len=24, feature_dim=3)
                generator = TimeGANGenerator(config)
                
                # Generate synthetic sample [1, 24, 3]
                # In production, we would load pre-trained weights here
                z = torch.randn(1, 24, 3) 
                synthetic_data = generator(z).detach().numpy()[0] # [24, 3]
                
                # Denormalize (Mocking the scaler logic)
                ghi = np.clip(synthetic_data[:, 0] * 1400, 0, 1400)
                wind = np.clip(synthetic_data[:, 1] * 30, 0, 30)
                temp = (synthetic_data[:, 2] * 40) + 10
                
                logger.info("   > Generated Synthetic Chaos Event via TimeGAN")
                
            except Exception as e:
                logger.error(f"   ❌ TimeGAN Failed: {e}. Falling back to standard generation.")
                ghi, wind, temp = self._generate_standard_profile(profile_type)
        
        # B. STANDARD SCENARIO (NREL Curves)
        else:
            ghi, wind, temp = self._generate_standard_profile(profile_type)

        # C. V2G STOCHASTIC MODELING
        # Generates EV fleet availability based on time-of-day probabilities
        v2g_profile = self._generate_v2g_availability()

        return {
            "time_step": np.linspace(0, 23, 24).tolist(),
            "ghi": ghi.tolist(),
            "wind_speed": wind.tolist(),
            "temperature": temp.tolist(),
            "v2g_availability_kw": v2g_profile.tolist() # New Data Source for Router
        }

    def _generate_standard_profile(self, profile_type: str):
        """Helper to generate standard Gaussian/Sinusoidal curves."""
        t = np.linspace(0, 23, 24)
        
        if "solar" in profile_type:
            # Solar Peak at 1 PM
            ghi = 1050 * np.exp(-0.5 * ((t - 13) / 3.5)**2)
            temp = 25 + (ghi / 50) + np.random.normal(0, 1, 24)
            wind = 5 + 2 * np.sin(t / 4) + np.random.normal(0, 1, 24)
        elif "wind" in profile_type:
            ghi = 600 * np.exp(-0.5 * ((t - 12) / 4)**2)
            wind = 12 + 8 * np.sin(t / 3) + np.random.normal(0, 3, 24)
            temp = 15 + np.random.normal(0, 2, 24)
        else:
            ghi = np.zeros(24)
            wind = np.zeros(24)
            temp = np.ones(24) * 20
            
        # Add sensor noise
        ghi = np.clip(ghi + np.random.normal(0, 20, 24), 0, 1400)
        return ghi, wind, temp

    def _generate_v2g_availability(self) -> np.ndarray:
        """
        Simulates availability of EV Fleet for Vehicle-to-Grid services.
        Logic:
        - 08:00 - 18:00 (Work): High availability (Office Parking)
        - 18:00 - 08:00 (Home): Lower availability (Residential charging priority)
        """
        t = np.linspace(0, 23, 24)
        
        # Base availability curve (High during day, low at night)
        # Sigmoid function to smooth transitions
        work_hours = 1 / (1 + np.exp(-(t - 8))) - 1 / (1 + np.exp(-(t - 18)))
        
        # Scale to kW (e.g., 20 cars * 7kW charger = ~140kW max)
        base_capacity = 140.0 
        availability = base_capacity * (0.3 + 0.7 * work_hours) 
        
        # Add stochastic uncertainty (drivers leaving early, traffic, etc.)
        noise = np.random.normal(0, 10, 24)
        availability = np.clip(availability + noise, 0, base_capacity)
        
        return availability

# Global Instance
data_manager = DataManager()
