"""
Multi-Source Active Energy Router.
Implements the Dispatch Hierarchy defined in the Integration Plan.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SourceState:
    solar_kw: float
    wind_kw: float
    battery_soc: float      # 0.0 to 1.0
    v2g_available_kw: float
    load_demand_kw: float
    diesel_status: str      # 'OFF', 'WARMUP', 'ACTIVE'

class EnergyRouter:
    def __init__(self):
        # Configuration from Strategy Docs
        self.BATTERY_MIN_SOC = 0.2
        self.BATTERY_MAX_POWER = 50.0  # kW
        self.CRITICAL_VOLTAGE = 0.94   # p.u.
        
    def compute_dispatch(self, state: SourceState, voltage_pu: float) -> Dict[str, Any]:
        """
        Decides active power setpoints based on Physics State (Voltage) and Asset State.
        """
        response = {
            "action": "IDLE",
            "battery_kw": 0.0, # +Charge, -Discharge
            "diesel_kw": 0.0,
            "v2g_kw": 0.0,
            "curtailment_kw": 0.0,
            "alert": "NOMINAL"
        }

        # 1. Physics Safety Override (The "Blind Spot" Protection)
        if voltage_pu < self.CRITICAL_VOLTAGE:
            response["alert"] = "CRITICAL_VOLTAGE_SUPPORT"
            response["action"] = "EMERGENCY_GENERATION"
            # Maximize all supports
            response["diesel_kw"] = 100.0 
            response["v2g_kw"] = -state.v2g_available_kw 
            if state.battery_soc > 0.1:
                response["battery_kw"] = -self.BATTERY_MAX_POWER
            return response

        # 2. Net Load Calculation
        renewable_gen = state.solar_kw + state.wind_kw
        net_load = state.load_demand_kw - renewable_gen

        # 3. Dispatch Logic
        if net_load > 0:
            # DEFICIT: Supply needed
            remaining = net_load
            
            # Priority A: Battery (Fastest Response)
            if state.battery_soc > self.BATTERY_MIN_SOC:
                discharge = min(remaining, self.BATTERY_MAX_POWER)
                response["battery_kw"] = -discharge
                remaining -= discharge
                response["action"] = "DISCHARGE_BESS"
            
            # Priority B: V2G (Community Support)
            if remaining > 0 and state.v2g_available_kw > 0:
                v2g_draw = min(remaining, state.v2g_available_kw)
                response["v2g_kw"] = -v2g_draw
                remaining -= v2g_draw
                response["action"] = "DISCHARGE_V2G"
            
            # Priority C: Diesel (Last Resort)
            if remaining > 0:
                response["diesel_kw"] = remaining
                response["action"] = "START_DIESEL"

        else:
            # SURPLUS: Store excess
            excess = abs(net_load)
            
            # Priority A: Battery Charge
            if state.battery_soc < 0.95:
                charge = min(excess, self.BATTERY_MAX_POWER)
                response["battery_kw"] = charge
                excess -= charge
                response["action"] = "CHARGE_BESS"
            
            # Priority B: V2G Charge
            if excess > 0:
                response["v2g_kw"] = excess
                response["action"] = "CHARGE_EVS"
            
            # Priority C: Curtailment (Avoid Over-Voltage)
            if excess > 0:
                response["curtailment_kw"] = excess

        return response
