"""
Severe Weather Resilience Analysis.
Simulates the impact of extreme meteorological events (e.g., 'Cloud Gorilla', Heatwaves)
on Microgrid Voltage Stability and Thermal Limits.

Synced via Jupytext for version control.
"""

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.control as control
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict

# --- Configuration & Physics Constants ---
@dataclass
class SimulationConfig:
    network_model: str = "cigre_mv_all"  # Standard Medium Voltage Benchmark
    base_mva: float = 10.0
    voltage_min_pu: float = 0.95
    voltage_max_pu: 1.05
    thermal_limit_pct: float = 100.0
    time_steps: int = 60  # 1 hour at 1-minute resolution

class WeatherScenarioGenerator:
    """
    Generates synthetic 'Black Swan' weather data for stress testing.
    Principle: Chaos Engineering (Inject faults to test resilience).
    """
    
    @staticmethod
    def generate_cloud_gorilla(duration_mins: int, magnitude_drop: float = 0.8) -> pd.DataFrame:
        """
        Simulates a massive, rapid cloud front (80% PV drop in < 5 mins).
        
        Args:
            duration_mins: Length of simulation.
            magnitude_drop: Severity of irradiance loss (0.0 - 1.0).
        """
        # Baseline: Clear sky (High Irradiance)
        t = np.linspace(0, duration_mins, duration_mins)
        irradiance = np.full_like(t, 1000.0) # 1000 W/m2
        
        # Event: Sudden drop at minute 10, recovery at minute 40
        start_event, end_event = 10, 40
        ramp_rate = 0.2 # Fast ramp
        
        # Apply physics-constrained drop
        mask = (t >= start_event) & (t <= end_event)
        irradiance[mask] = 1000.0 * (1.0 - magnitude_drop)
        
        # Add slight noise (Sensor uncertainty)
        noise = np.random.normal(0, 10, duration_mins)
        irradiance = np.clip(irradiance + noise, 0, 1400) # Clip to physical limits
        
        return pd.DataFrame({"time_step": t, "ghi_w_m2": irradiance})

class PhysicsEngine:
    """
    Wraps Pandapower for High-Fidelity Power Flow simulations.
    Principle: Single Responsibility (Manages Grid Physics).
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        # Initialize grid topology
        # In prod, this loads from the 'grid_parameters.json' file
        self.net = pn.create_cigre_network_mv(with_der="pv_wind")
        
        # Fail Fast: Verify network converged initially
        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            raise RuntimeError("CRITICAL: Base network topology is unstable!")

    def run_time_series(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes Quasi-Static Time Series (QSTS) simulation.
        """
        results = []
        
        # Identify PV buses
        pv_indices = self.net.sgen[self.net.sgen.type == "PV"].index
        
        for _, row in weather_data.iterrows():
            ghi = row["ghi_w_m2"]
            
            # Physics: Update PV generation based on Irradiance
            # P_gen = Efficiency * Area * Irradiance (Simplified linear model)
            # We scale the nominal power of PVs in the grid
            scaling_factor = ghi / 1000.0
            self.net.sgen.loc[pv_indices, "scaling"] = scaling_factor
            
            try:
                pp.runpp(self.net, numba=False) # Numba disabled for stability in scripts
                
                # Capture Grid State
                state_snapshot = {
                    "time": row["time_step"],
                    "min_voltage_pu": self.net.res_bus.vm_pu.min(),
                    "max_line_loading_pct": self.net.res_line.loading_percent.max(),
                    "total_losses_mw": self.net.res_line.pl_mw.sum()
                }
                results.append(state_snapshot)
                
            except pp.LoadflowNotConverged:
                # Capture Collapse
                results.append({
                    "time": row["time_step"],
                    "min_voltage_pu": 0.0, # Collapse
                    "max_line_loading_pct": 999.9,
                    "total_losses_mw": 0.0
                })
        
        return pd.DataFrame(results)

def analyze_results(results: pd.DataFrame, config: SimulationConfig):
    """
    Visualizes the stress test outcomes.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Voltage Stability
    color = 'tab:blue'
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Min Voltage (p.u.)', color=color)
    ax1.plot(results["time"], results["min_voltage_pu"], color=color, linewidth=2, label="Grid Voltage")
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add Limit Lines
    ax1.axhline(y=config.voltage_min_pu, color='r', linestyle='--', label="Under-Voltage Limit")

    # Plot Thermal Loading
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Max Line Loading (%)', color=color)
    ax2.plot(results["time"], results["max_line_loading_pct"], color=color, linestyle='-.', label="Thermal Load")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Microgrid Resilience Analysis: Cloud Gorilla Event")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Save for reporting
    plt.savefig("severe_weather_impact.png")
    print("✅ Analysis Complete. Plot saved to 'severe_weather_impact.png'.")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup
    config = SimulationConfig()
    engine = PhysicsEngine(config)
    
    # 2. Generate Black Swan Scenario
    print("⚡ Generating 'Cloud Gorilla' Weather Front...")
    weather_df = WeatherScenarioGenerator.generate_cloud_gorilla(duration_mins=60)
    
    # 3. Run Physics Simulation
    print(f"🔄 Simulating Grid Dynamics on {config.network_model}...")
    sim_results = engine.run_time_series(weather_df)
    
    # 4. Analyze
    violations = sim_results[sim_results["min_voltage_pu"] < config.voltage_min_pu]
    if not violations.empty:
        print(f"⚠️ ALERT: Voltage violations detected in {len(violations)} time steps!")
    else:
        print("✅ Grid remained stable throughout the event.")
        
    analyze_results(sim_results, config)
