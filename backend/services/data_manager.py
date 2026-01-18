"""
Data Manager Service.
Central repository for 'Valuable Inputs' (Grid Topologies & Weather Contexts).

Responsible for:
1. Loading standard IEEE Benchmarks (14, 30, 118 bus) via Pandapower.
2. Converting physical grids into Graph representations (Adjacency Matrices) for DeepONet.
3. Generating synthetic 'NREL-like' weather profiles for rapid prototyping.
"""

import logging
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
from typing import Dict, Any, List, Optional

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
        Loads a standard IEEE grid and converts it to a graph representation
        suitable for the Physics-Informed DeepONet.
        
        Args:
            grid_name: 'ieee14', 'ieee30', or 'ieee118'
            
        Returns:
            Dictionary containing:
            - n_buses: Total number of nodes (Critical for DeepONet resizing)
            - adj_matrix: Dense adjacency matrix (Connections)
            - bus_features: Static features (Voltage limits)
            - line_params: Physics parameters (Resistance/Reactance) for Loss calculation
        """
        grid_name = grid_name.lower().strip()
        
        if grid_name not in self.SUPPORTED_GRIDS:
            raise ValueError(f"Grid '{grid_name}' not supported. Available: {self.SUPPORTED_GRIDS}")

        logger.info(f"🔌 Loading Grid Topology: {grid_name.upper()}...")

        # 1. Load Standard Network from Pandapower
        if grid_name == "ieee14":
            net = pn.case14()
        elif grid_name == "ieee30":
            net = pn.case30()
        elif grid_name == "ieee118":
            net = pn.case118()
            
        # 2. Construct Graph Representation (Physics Core)
        # We need an Adjacency Matrix A where A[i,j] = 1 if connected
        # DeepONet uses this to understand the 'Trunk' (Location) space.
        
        # Create NetworkX graph from lines
        # MultiGraph=False ensures we flatten parallel lines for the simple adjacency
        mg = pp.topology.create_nxgraph(net, include_lines=True, include_trafo=True, multi=False)
        
        # Get Adjacency Matrix (Sorted by Bus ID to ensure alignment)
        # This maps the physical topology to the Neural Network's input
        adj_matrix = nx.to_numpy_array(mg, nodelist=sorted(mg.nodes()))
        
        # 3. Extract Physics Parameters (for Constraints/Loss)
        # Resistance (R) and Reactance (X) are needed for the Power Flow Equation
        n_lines = len(net.line)
        r_ohm = net.line['r_ohm_per_km'].values
        x_ohm = net.line['x_ohm_per_km'].values
        
        # 4. Extract Bus Constraints (Safety Limits)
        # Used by the Projection Layer to clamp predictions
        n_buses = len(net.bus)
        vm_min = net.bus['min_vm_pu'].fillna(0.9).values # Default IEEE lower bound
        vm_max = net.bus['max_vm_pu'].fillna(1.1).values # Default IEEE upper bound
        
        topology_data = {
            "name": grid_name,
            "n_buses": int(n_buses),  # <--- Input for DeepONet __init__
            "n_lines": int(n_lines),
            "adj_matrix": adj_matrix, # <--- Input for Graph Convolutions (if added later)
            "bus_constraints": {
                "vm_min": vm_min,
                "vm_max": vm_max
            },
            "physics_params": {
                "r_lines": r_ohm,
                "x_lines": x_ohm
            }
        }
        
        logger.info(f"   > Loaded {n_buses} Buses and {n_lines} Lines.")
        return topology_data

    def get_weather_profile(self, profile_type: str = "solar_egypt") -> Dict[str, List[float]]:
        """
        Generates a synthetic 24-hour weather profile mimicking NREL datasets.
        Used to feed the 'Branch Net' (Input Function) of the DeepONet.
        
        Args:
            profile_type: 'solar_egypt' (High GHI), 'wind_north' (High Wind), 'storm' (Chaos)
            
        Returns:
            Dictionary with 24-step time-series for GHI, Wind, Temp.
        """
        logger.info(f"☀️  Generating Weather Context: {profile_type}...")
        
        # Time axis: 0 to 23 hours
        t = np.linspace(0, 23, 24)
        
        # Base Profiles
        if "solar" in profile_type:
            # Solar: Gaussian Curve peaking at 1 PM (t=13)
            # Peak irradiance ~1050 W/m2 (Egypt Summer)
            ghi = 1050 * np.exp(-0.5 * ((t - 13) / 3.5)**2)
            
            # Temperature: Follows solar with lag
            temp = 25 + (ghi / 50) + np.random.normal(0, 1, 24)
            
            # Wind: Moderate breeze
            wind = 5 + 2 * np.sin(t / 4) + np.random.normal(0, 1, 24)
            
        elif "wind" in profile_type:
            # Solar: Weaker (Winter/Cloudy)
            ghi = 600 * np.exp(-0.5 * ((t - 12) / 4)**2)
            
            # Wind: Stronger, more volatile
            wind = 12 + 8 * np.sin(t / 3) + np.random.normal(0, 3, 24)
            wind = np.clip(wind, 0, 30) # Max wind speed
            
            temp = 15 + np.random.normal(0, 2, 24)
            
        else:
            # Default / Fallback
            ghi = np.zeros(24)
            wind = np.zeros(24)
            temp = np.ones(24) * 20

        # Add stochastic noise (Sensor noise simulation)
        ghi = np.clip(ghi + np.random.normal(0, 20, 24), 0, 1400)
        
        return {
            "time_step": t.tolist(),
            "ghi": ghi.tolist(),          # Global Horizontal Irradiance (W/m^2)
            "wind_speed": wind.tolist(),  # m/s
            "temperature": temp.tolist()  # Celsius
        }

# Global Instance for Singleton Access in API
data_manager = DataManager()

# --- Unit Test (Manual Verification) ---
if __name__ == "__main__":
    # Simulate User Input logic
    test_grid = "ieee118"
    
    try:
        data = data_manager.get_topology(test_grid)
        print(f"\n✅ Successfully Loaded {test_grid.upper()}")
        print(f"   Buses: {data['n_buses']}")
        print(f"   Adjacency Shape: {data['adj_matrix'].shape}")
        
        weather = data_manager.get_weather_profile("solar_egypt")
        print(f"✅ Generated Weather: {len(weather['ghi'])} time steps")
        print(f"   Peak Solar: {max(weather['ghi']):.2f} W/m^2")
        
    except Exception as e:
        print(f"❌ Error: {e}")
