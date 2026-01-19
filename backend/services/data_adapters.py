"""
Data Adapters for Asset Simulation.
Handles the physical and behavioral constraints of non-deterministic assets.

1. V2GAdapter: Models EV Fleet availability (Stochastic).
2. DieselAdapter: Models Generator state machine (Physical constraints).
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AssetAdapters")

@dataclass
class DieselState:
    status: str = "OFF"       # OFF, WARMUP, ACTIVE, COOLDOWN
    warmup_counter: int = 0   # Minutes remaining
    cooldown_counter: int = 0 # Minutes remaining
    active_power_kw: float = 0.0

class DieselAdapter:
    """
    State Machine for Diesel Generators.
    Enforces physical delays:
    - Warmup: 5 minutes to reach synchronization speed.
    - Cooldown: 5 minutes idle before shutdown to prevent thermal shock.
    """
    def __init__(self, max_power_kw: float = 500.0, warmup_min: int = 5, cooldown_min: int = 5):
        self.max_kw = max_power_kw
        self.warmup_time = warmup_min
        self.cooldown_time = cooldown_min
        self.state = DieselState()

    def update(self, request_start: bool, load_kw: float) -> DieselState:
        """
        Advances the generator state by one time step (assumed 1 minute for logic).
        """
        s = self.state
        
        # State Machine Logic
        if s.status == "OFF":
            s.active_power_kw = 0.0
            if request_start:
                s.status = "WARMUP"
                s.warmup_counter = self.warmup_time
                logger.info("⚙️ Diesel Generator Starting (Warmup Phase)...")

        elif s.status == "WARMUP":
            s.active_power_kw = 0.0
            if s.warmup_counter > 0:
                s.warmup_counter -= 1
            else:
                s.status = "ACTIVE"
                logger.info("✅ Diesel Generator Online (Synchronized).")

        elif s.status == "ACTIVE":
            s.active_power_kw = min(load_kw, self.max_kw)
            if not request_start:
                s.status = "COOLDOWN"
                s.cooldown_counter = self.cooldown_time
                logger.info("❄️ Diesel Generator Stopping (Cooldown Phase)...")

        elif s.status == "COOLDOWN":
            s.active_power_kw = 0.0 # Typically idles without load
            if s.cooldown_counter > 0:
                s.cooldown_counter -= 1
            else:
                s.status = "OFF"
                logger.info("🛑 Diesel Generator Offline.")

        return s

class V2GAdapter:
    """
    Stochastic Model for Vehicle-to-Grid Availability.
    Simulates a fleet of 50 EVs (Nissan Leaf / Tesla Model 3 mix).
    """
    def __init__(self, fleet_size: int = 50, avg_charger_kw: float = 7.0):
        self.max_capacity = fleet_size * avg_charger_kw # ~350 kW theoretical max

    def get_availability(self, hour_of_day: int) -> float:
        """
        Returns available kW capacity based on time of day.
        Logic:
        - 09:00 - 17:00: Work Mode (High availability at office chargers).
        - 18:00 - 07:00: Home Mode (Lower V2G participation due to user anxiety).
        """
        # Base Probability Curve (Sigmoidal transitions)
        if 8 <= hour_of_day <= 18:
            # Office Hours: 80% of fleet plugged in
            base_prob = 0.8
        else:
            # Night/Home: 30% of fleet willing to discharge
            base_prob = 0.3
            
        # Add Stochastic Noise (Traffic, Holidays, etc.)
        noise = np.random.normal(0, 0.05)
        participation_rate = np.clip(base_prob + noise, 0.1, 1.0)
        
        available_kw = self.max_capacity * participation_rate
        return round(available_kw, 2)

# --- Unit Test ---
if __name__ == "__main__":
    print("Testing Adapters...")
    
    # Test 1: V2G
    v2g = V2GAdapter()
    print(f"V2G Capacity (12:00 PM): {v2g.get_availability(12)} kW")
    print(f"V2G Capacity (03:00 AM): {v2g.get_availability(3)} kW")
    
    # Test 2: Diesel
    gen = DieselAdapter(warmup_min=2)
    print("\nDiesel Start Sequence:")
    print(f"T=0: {gen.update(True, 100).status}")   # WARMUP
    print(f"T=1: {gen.update(True, 100).status}")   # WARMUP
    print(f"T=2: {gen.update(True, 100).status}")   # ACTIVE
    print(f"T=3: Power {gen.update(True, 100).active_power_kw} kW") # 100 kW
