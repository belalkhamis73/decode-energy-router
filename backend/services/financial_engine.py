"""
Financial & Environmental KPI Engine.
Calculates the 'Value' of the energy routing decisions in real-time.

Responsibility:
Translates physical dispatch actions (kW) into business metrics (USD, kgCO2).
Compares the 'Active Router' performance against a 'Do Nothing' baseline (Grid + Diesel).
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FinancialConfig:
    """
    Economic & Environmental Constants.
    In production, these would be loaded from a live market API or user config.
    """
    # Costs (USD/kWh)
    GRID_COST_PEAK = 0.25      # Expensive during high demand
    GRID_COST_OFF_PEAK = 0.10
    DIESEL_LCOE = 0.45         # High OpEx due to fuel + maintenance
    SOLAR_LCOE = 0.03          # Very low marginal cost
    BATTERY_CYCLE_COST = 0.05  # Cost of degradation per kWh throughput
    V2G_REWARD = 0.15          # Incentive paid to EV owners for discharging

    # Carbon Intensity (kgCO2/kWh)
    GRID_CARBON = 0.45         # Natural Gas / Mixed Grid average
    DIESEL_CARBON = 0.85       # High emissions
    SOLAR_CARBON = 0.0         # Clean

class FinancialAnalyzer:
    def calculate_metrics(self, dispatch: Dict[str, Any], load_demand_kw: float, tick: int) -> Dict[str, float]:
        """
        Computes OpEx savings and CO2 reduction compared to a 'Business as Usual' baseline.
        
        Args:
            dispatch: The output from EnergyRouter (contains battery_kw, diesel_kw, etc.)
            load_demand_kw: The total facility load at this timestamp.
            tick: Hour of day (0-23) to determine Peak/Off-Peak pricing.
            
        Returns:
            Dictionary of financial KPIs for the dashboard.
        """
        # 1. Determine Current Grid Price (Time-of-Use)
        # Peak hours assumed 17:00 - 21:00
        grid_price = self.GRID_COST_PEAK if 17 <= tick <= 21 else self.GRID_COST_OFF_PEAK

        # 2. Calculate 'Actual' System Cost (With D.E.C.O.D.E. Router)
        # Cost = (Solar * LCOE) + (Diesel * LCOE) + (Battery_Discharge * Wear) + (Grid_Import * Price)
        
        # Note: dispatch['battery_kw'] is negative for discharge. We take abs() for cost calculation.
        battery_throughput = abs(dispatch.get('battery_kw', 0.0))
        diesel_gen = dispatch.get('diesel_kw', 0.0)
        v2g_flow = dispatch.get('v2g_kw', 0.0) # +Charge, -Discharge
        
        # Grid Import = Load - (Solar + Wind + Diesel + Battery_Discharge + V2G_Discharge)
        # Simplified: Anything not met by onsite resources is Grid Import
        # (This is an approximation for the KPI engine)
        onsite_supply = (
            dispatch.get('solar_kw', 0.0) + 
            dispatch.get('wind_kw', 0.0) + 
            diesel_gen + 
            max(0, -dispatch.get('battery_kw', 0.0)) + # Discharging adds supply
            max(0, -v2g_flow)
        )
        
        grid_import_actual = max(0, load_demand_kw - onsite_supply)
        
        cost_actual = (
            (dispatch.get('solar_kw', 0.0) * self.SOLAR_LCOE) +
            (diesel_gen * self.DIESEL_LCOE) +
            (battery_throughput * self.BATTERY_CYCLE_COST) +
            (grid_import_actual * grid_price)
        )
        
        # Add V2G Incentive Cost (We pay users to discharge)
        if v2g_flow < 0:
            cost_actual += abs(v2g_flow) * self.V2G_REWARD

        # 3. Calculate 'Baseline' Cost (Legacy System)
        # Assumption: Without Router, load is met 100% by Grid, or Diesel during outages.
        # Here we assume standard operation: 100% Grid Import.
        cost_baseline = load_demand_kw * grid_price

        # 4. Carbon Footprint Calculation
        carbon_actual = (
            (grid_import_actual * self.GRID_CARBON) +
            (diesel_gen * self.DIESEL_CARBON)
        )
        carbon_baseline = load_demand_kw * self.GRID_CARBON

        # 5. Delta (The Value Proposition)
        metrics = {
            "opex_hourly_usd": round(cost_actual, 2),
            "baseline_hourly_usd": round(cost_baseline, 2),
            "savings_usd": round(cost_baseline - cost_actual, 2),
            "co2_emitted_kg": round(carbon_actual, 2),
            "co2_avoided_kg": round(carbon_baseline - carbon_actual, 2),
            "roi_factor": round((cost_baseline - cost_actual) / max(1.0, cost_actual) * 100, 1) # ROI %
        }
        
        return metrics

# --- Unit Test ---
if __name__ == "__main__":
    analyzer = FinancialAnalyzer()
    
    # Mock Scenario: Peak Hour, Heavy Load, Battery Discharging to save money
    mock_dispatch = {
        "solar_kw": 50.0,
        "wind_kw": 10.0,
        "battery_kw": -20.0, # Discharging
        "diesel_kw": 0.0,
        "v2g_kw": 0.0
    }
    
    kpis = analyzer.calculate_metrics(mock_dispatch, load_demand_kw=100.0, tick=18)
    print("FINANCIAL REPORT (Unit Test):")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
