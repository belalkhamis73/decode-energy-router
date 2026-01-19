
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SourceState:
    solar_kw: float; wind_kw: float; battery_soc: float; v2g_available_kw: float; load_demand_kw: float; diesel_status: str

class EnergyRouter:
    def compute_dispatch(self, state: SourceState, voltage_pu: float, stability_score: float = 1.0) -> Dict[str, Any]:
        net = state.load_demand_kw - (state.solar_kw + state.wind_kw)
        resp = {"action": "IDLE", "battery_kw": 0.0, "diesel_kw": 0.0}
        
        if voltage_pu < 0.92:
            resp["action"] = "START_DIESEL"; resp["diesel_kw"] = 100.0
            return resp

        if net > 0:
            if state.battery_soc > 0.2:
                resp["battery_kw"] = -min(net, 50.0); resp["action"] = "DISCHARGE_BESS"
            else:
                resp["diesel_kw"] = net; resp["action"] = "START_DIESEL"
        else:
            resp["battery_kw"] = min(abs(net), 50.0); resp["action"] = "CHARGE_BESS"
        return resp
