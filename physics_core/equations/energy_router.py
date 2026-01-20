"""
physics_core/equations/energy_router.py

Enhanced energy routing with multi-objective optimization, constraint prioritization,
and decision tree logging for microgrid dispatch.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import math


class ConstraintPriority(Enum):
    """Hierarchical constraint priorities"""
    SAFETY = 1  # Voltage/frequency stability, thermal limits
    OPERATIONAL = 2  # SOC reserves, equipment limits
    USER = 3  # User-defined preferences
    ECONOMIC = 4  # Cost optimization


class ActionType(Enum):
    """Possible dispatch actions"""
    IDLE = "IDLE"
    CHARGE_BESS = "CHARGE_BESS"
    DISCHARGE_BESS = "DISCHARGE_BESS"
    START_DIESEL = "START_DIESEL"
    STOP_DIESEL = "STOP_DIESEL"
    ENABLE_V2G = "ENABLE_V2G"
    LOAD_SHED = "LOAD_SHED"
    EXPORT_GRID = "EXPORT_GRID"


@dataclass
class SourceState:
    """Current state of energy sources"""
    solar_kw: float
    wind_kw: float
    battery_soc: float  # 0.0 to 1.0
    v2g_available_kw: float
    load_demand_kw: float
    diesel_status: str  # "OFF", "WARMING", "RUNNING"
    battery_temp_c: float = 25.0
    diesel_runtime_hours: float = 0.0


@dataclass
class Constraints:
    """System constraints and limits"""
    # Safety constraints (Priority 1)
    voltage_min_pu: float = 0.92
    voltage_max_pu: float = 1.08
    frequency_min_hz: float = 59.5
    frequency_max_hz: float = 60.5
    battery_temp_max_c: float = 45.0
    
    # Operational constraints (Priority 2)
    battery_soc_min: float = 0.2
    battery_soc_max: float = 0.95
    battery_power_max_kw: float = 50.0
    diesel_min_load_kw: float = 20.0
    diesel_max_kw: float = 100.0
    
    # User constraints (Priority 3)
    user_soc_reserve: float = 0.3
    avoid_diesel: bool = False
    allow_v2g: bool = True


@dataclass
class Forecast:
    """15-minute lookahead forecast"""
    solar_kw: List[float] = field(default_factory=list)
    wind_kw: List[float] = field(default_factory=list)
    load_kw: List[float] = field(default_factory=list)
    timestep_minutes: float = 5.0


@dataclass
class UserPreferences:
    """User-defined optimization preferences"""
    cost_weight: float = 1.0  # Weight for economic optimization
    battery_wear_weight: float = 0.5  # Penalty for battery cycling
    diesel_cost_per_kwh: float = 0.30
    grid_import_cost_per_kwh: float = 0.12
    grid_export_price_per_kwh: float = 0.08
    prefer_renewables: bool = True


@dataclass
class DecisionNode:
    """Node in decision tree explaining routing choice"""
    constraint: str
    priority: ConstraintPriority
    active: bool
    value: float
    limit: float
    message: str
    satisfied: bool


@dataclass
class RoutingDecision:
    """Complete dispatch decision with explanation"""
    action: ActionType
    battery_kw: float  # Positive = charge, negative = discharge
    diesel_kw: float
    v2g_kw: float
    grid_kw: float  # Positive = import, negative = export
    
    # Decision explanation
    decision_tree: List[DecisionNode]
    active_constraints: List[str]
    opportunity_cost: float
    objective_score: float
    
    # Multi-step plan
    future_actions: List[Tuple[float, str]]  # (time_minutes, action_description)


class EnergyRouter:
    """
    Enhanced energy router with multi-objective optimization.
    
    Maintains backward compatibility with compute_dispatch() while adding
    advanced v2 methods for constraint-aware optimization.
    """
    
    def __init__(self):
        self.decision_history: List[RoutingDecision] = []
        self.simulation_timestep = 1.0  # minutes
    
    # === LEGACY COMPATIBILITY ===
    
    def compute_dispatch(self, state: SourceState, voltage_pu: float, 
                        stability_score: float = 1.0) -> Dict[str, Any]:
        """
        Legacy method - maintains backward compatibility.
        Simple dispatch without advanced optimization.
        """
        net = state.load_demand_kw - (state.solar_kw + state.wind_kw)
        resp = {"action": "IDLE", "battery_kw": 0.0, "diesel_kw": 0.0}
        
        if voltage_pu < 0.92:
            resp["action"] = "START_DIESEL"
            resp["diesel_kw"] = 100.0
            return resp

        if net > 0:
            if state.battery_soc > 0.2:
                resp["battery_kw"] = -min(net, 50.0)
                resp["action"] = "DISCHARGE_BESS"
            else:
                resp["diesel_kw"] = net
                resp["action"] = "START_DIESEL"
        else:
            resp["battery_kw"] = min(abs(net), 50.0)
            resp["action"] = "CHARGE_BESS"
        return resp
    
    # === ENHANCED V2 METHODS ===
    
    def compute_dispatch_v2(
        self,
        sources: SourceState,
        constraints: Constraints,
        forecasts: Optional[Forecast] = None,
        user_preferences: Optional[UserPreferences] = None,
        voltage_pu: float = 1.0,
        frequency_hz: float = 60.0
    ) -> RoutingDecision:
        """
        Multi-objective optimization with constraint prioritization.
        
        Solves dispatch in priority order:
        1. Safety constraints (voltage, frequency, thermal)
        2. Operational constraints (SOC limits, equipment ratings)
        3. User preferences (SOC reserves, diesel avoidance)
        4. Economic optimization (minimize cost)
        """
        if user_preferences is None:
            user_preferences = UserPreferences()
        
        decision_tree: List[DecisionNode] = []
        
        # Calculate net load
        renewable_kw = sources.solar_kw + sources.wind_kw
        net_load_kw = sources.load_demand_kw - renewable_kw
        
        # Initialize decision
        battery_kw = 0.0
        diesel_kw = 0.0
        v2g_kw = 0.0
        grid_kw = 0.0
        action = ActionType.IDLE
        
        # === PRIORITY 1: SAFETY CONSTRAINTS ===
        
        # Check voltage stability
        voltage_ok = self._check_constraint(
            decision_tree,
            "Voltage Stability",
            ConstraintPriority.SAFETY,
            voltage_pu,
            constraints.voltage_min_pu,
            constraints.voltage_max_pu
        )
        
        if not voltage_ok:
            # Emergency response: start diesel immediately
            diesel_kw = min(constraints.diesel_max_kw, sources.load_demand_kw * 0.5)
            action = ActionType.START_DIESEL
            decision_tree.append(DecisionNode(
                constraint="Emergency Voltage Support",
                priority=ConstraintPriority.SAFETY,
                active=True,
                value=voltage_pu,
                limit=constraints.voltage_min_pu,
                message=f"Voltage {voltage_pu:.3f} pu below minimum, starting diesel",
                satisfied=False
            ))
            return self._build_decision(
                action, battery_kw, diesel_kw, v2g_kw, grid_kw,
                decision_tree, sources, constraints, user_preferences
            )
        
        # Check frequency stability
        freq_ok = self._check_constraint(
            decision_tree,
            "Frequency Stability",
            ConstraintPriority.SAFETY,
            frequency_hz,
            constraints.frequency_min_hz,
            constraints.frequency_max_hz
        )
        
        # Check battery temperature
        temp_ok = sources.battery_temp_c < constraints.battery_temp_max_c
        decision_tree.append(DecisionNode(
            constraint="Battery Temperature",
            priority=ConstraintPriority.SAFETY,
            active=True,
            value=sources.battery_temp_c,
            limit=constraints.battery_temp_max_c,
            message=f"Battery temp {sources.battery_temp_c:.1f}°C",
            satisfied=temp_ok
        ))
        
        # === PRIORITY 2: OPERATIONAL CONSTRAINTS ===
        
        # Available battery capacity
        if net_load_kw > 0:  # Need to discharge
            max_discharge_kw = min(
                constraints.battery_power_max_kw,
                (sources.battery_soc - constraints.battery_soc_min) * 100.0  # Assume 100kWh nominal
            )
            max_discharge_kw = max(0, max_discharge_kw)
        else:  # Can charge
            max_charge_kw = min(
                constraints.battery_power_max_kw,
                (constraints.battery_soc_max - sources.battery_soc) * 100.0
            )
            max_charge_kw = max(0, max_charge_kw)
        
        # === PRIORITY 3: USER CONSTRAINTS ===
        
        user_soc_ok = sources.battery_soc > constraints.user_soc_reserve
        decision_tree.append(DecisionNode(
            constraint="User SOC Reserve",
            priority=ConstraintPriority.USER,
            active=True,
            value=sources.battery_soc,
            limit=constraints.user_soc_reserve,
            message=f"SOC {sources.battery_soc:.1%} vs reserve {constraints.user_soc_reserve:.1%}",
            satisfied=user_soc_ok
        ))
        
        # === PRIORITY 4: ECONOMIC OPTIMIZATION ===
        
        if net_load_kw > 0:
            # Need power - evaluate options with lookahead
            options = self._evaluate_supply_options(
                net_load_kw, sources, constraints, user_preferences,
                forecasts, temp_ok, user_soc_ok
            )
            
            # Select best option
            best_option = min(options, key=lambda x: x['total_cost'])
            battery_kw = best_option.get('battery_kw', 0.0)
            diesel_kw = best_option.get('diesel_kw', 0.0)
            v2g_kw = best_option.get('v2g_kw', 0.0)
            grid_kw = best_option.get('grid_kw', 0.0)
            action = best_option['action']
            
        else:
            # Excess power - charge battery or export
            excess_kw = abs(net_load_kw)
            
            if sources.battery_soc < constraints.battery_soc_max and temp_ok:
                battery_kw = min(excess_kw, max_charge_kw)
                action = ActionType.CHARGE_BESS
                
                remaining = excess_kw - battery_kw
                if remaining > 1.0:
                    grid_kw = -remaining  # Export excess
                    action = ActionType.CHARGE_BESS  # Primary action is charging
            else:
                grid_kw = -excess_kw  # Export all
                action = ActionType.EXPORT_GRID
        
        return self._build_decision(
            action, battery_kw, diesel_kw, v2g_kw, grid_kw,
            decision_tree, sources, constraints, user_preferences, forecasts
        )
    
    def _evaluate_supply_options(
        self,
        needed_kw: float,
        sources: SourceState,
        constraints: Constraints,
        prefs: UserPreferences,
        forecasts: Optional[Forecast],
        temp_ok: bool,
        user_soc_ok: bool
    ) -> List[Dict[str, Any]]:
        """Evaluate different ways to meet power demand"""
        options = []
        
        # Option 1: Battery only (if available and allowed)
        if user_soc_ok and temp_ok:
            max_batt = min(
                constraints.battery_power_max_kw,
                (sources.battery_soc - constraints.battery_soc_min) * 100.0
            )
            if max_batt >= needed_kw:
                options.append({
                    'action': ActionType.DISCHARGE_BESS,
                    'battery_kw': -needed_kw,
                    'diesel_kw': 0.0,
                    'v2g_kw': 0.0,
                    'grid_kw': 0.0,
                    'total_cost': needed_kw * 0.02 * prefs.battery_wear_weight  # Battery wear
                })
        
        # Option 2: Battery + Diesel
        if not constraints.avoid_diesel:
            batt_contribution = min(
                constraints.battery_power_max_kw * 0.7,  # Reserve some capacity
                (sources.battery_soc - constraints.battery_soc_min) * 100.0
            ) if user_soc_ok and temp_ok else 0.0
            
            diesel_needed = max(needed_kw - batt_contribution, constraints.diesel_min_load_kw)
            
            if diesel_needed <= constraints.diesel_max_kw:
                diesel_cost = diesel_needed * prefs.diesel_cost_per_kwh
                batt_cost = batt_contribution * 0.02 * prefs.battery_wear_weight
                
                options.append({
                    'action': ActionType.START_DIESEL,
                    'battery_kw': -batt_contribution,
                    'diesel_kw': diesel_needed,
                    'v2g_kw': 0.0,
                    'grid_kw': 0.0,
                    'total_cost': (diesel_cost + batt_cost) * prefs.cost_weight
                })
        
        # Option 3: V2G (if available and allowed)
        if constraints.allow_v2g and sources.v2g_available_kw > 0:
            v2g_kw = min(needed_kw, sources.v2g_available_kw)
            remaining = needed_kw - v2g_kw
            
            options.append({
                'action': ActionType.ENABLE_V2G,
                'battery_kw': 0.0,
                'diesel_kw': remaining if remaining > 0 else 0.0,
                'v2g_kw': v2g_kw,
                'grid_kw': 0.0,
                'total_cost': v2g_kw * 0.05  # V2G compensation
            })
        
        # Option 4: Grid import (always available as fallback)
        options.append({
            'action': ActionType.IDLE,  # Grid is baseline
            'battery_kw': 0.0,
            'diesel_kw': 0.0,
            'v2g_kw': 0.0,
            'grid_kw': needed_kw,
            'total_cost': needed_kw * prefs.grid_import_cost_per_kwh * prefs.cost_weight
        })
        
        return options
    
    def _check_constraint(
        self,
        tree: List[DecisionNode],
        name: str,
        priority: ConstraintPriority,
        value: float,
        min_limit: float,
        max_limit: float
    ) -> bool:
        """Check if value is within limits and log to decision tree"""
        satisfied = min_limit <= value <= max_limit
        tree.append(DecisionNode(
            constraint=name,
            priority=priority,
            active=True,
            value=value,
            limit=f"{min_limit}-{max_limit}",
            message=f"{name}: {value:.3f} (limits: {min_limit:.3f}-{max_limit:.3f})",
            satisfied=satisfied
        ))
        return satisfied
    
    def _build_decision(
        self,
        action: ActionType,
        battery_kw: float,
        diesel_kw: float,
        v2g_kw: float,
        grid_kw: float,
        tree: List[DecisionNode],
        sources: SourceState,
        constraints: Constraints,
        prefs: UserPreferences,
        forecasts: Optional[Forecast] = None
    ) -> RoutingDecision:
        """Build complete routing decision with explanation"""
        
        # Calculate opportunity cost
        opp_cost = self.calculate_opportunity_cost(
            battery_kw, diesel_kw, grid_kw, prefs
        )
        
        # Calculate objective score
        obj_score = self._calculate_objective(
            battery_kw, diesel_kw, v2g_kw, grid_kw, prefs
        )
        
        # Get active constraints
        active = [node.constraint for node in tree if node.active and not node.satisfied]
        
        # Generate future plan if forecast available
        future_actions = []
        if forecasts and len(forecasts.load_kw) > 0:
            future_actions = self._plan_future_actions(
                sources, constraints, forecasts, prefs
            )
        
        decision = RoutingDecision(
            action=action,
            battery_kw=battery_kw,
            diesel_kw=diesel_kw,
            v2g_kw=v2g_kw,
            grid_kw=grid_kw,
            decision_tree=tree,
            active_constraints=active,
            opportunity_cost=opp_cost,
            objective_score=obj_score,
            future_actions=future_actions
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _plan_future_actions(
        self,
        sources: SourceState,
        constraints: Constraints,
        forecasts: Forecast,
        prefs: UserPreferences
    ) -> List[Tuple[float, str]]:
        """Plan next 15 minutes of actions based on forecast"""
        plan = []
        simulated_soc = sources.battery_soc
        time = 0.0
        
        for i, (solar, wind, load) in enumerate(zip(
            forecasts.solar_kw,
            forecasts.wind_kw,
            forecasts.load_kw
        )):
            time += forecasts.timestep_minutes
            net = load - (solar + wind)
            
            if net > 0 and simulated_soc < constraints.battery_soc_min:
                plan.append((time, f"Start diesel (~{net:.1f} kW needed)"))
            elif net < -20 and simulated_soc < constraints.battery_soc_max * 0.9:
                plan.append((time, f"Charge battery from {abs(net):.1f} kW excess"))
                simulated_soc = min(1.0, simulated_soc + 0.05)
            elif net > 0:
                simulated_soc = max(0.0, simulated_soc - 0.03)
        
        return plan[:3]  # Return next 3 key actions
    
    def _calculate_objective(
        self,
        battery_kw: float,
        diesel_kw: float,
        v2g_kw: float,
        grid_kw: float,
        prefs: UserPreferences
    ) -> float:
        """Calculate overall objective function value"""
        cost = 0.0
        cost += abs(battery_kw) * 0.02 * prefs.battery_wear_weight
        cost += diesel_kw * prefs.diesel_cost_per_kwh * prefs.cost_weight
        cost += max(0, grid_kw) * prefs.grid_import_cost_per_kwh * prefs.cost_weight
        cost -= max(0, -grid_kw) * prefs.grid_export_price_per_kwh * prefs.cost_weight
        return cost
    
    def explain_routing_decision(
        self,
        decision: Optional[RoutingDecision] = None
    ) -> str:
        """
        Returns human-readable explanation of routing decision.
        """
        if decision is None:
            if not self.decision_history:
                return "No decisions made yet"
            decision = self.decision_history[-1]
        
        explanation = []
        explanation.append(f"=== ROUTING DECISION: {decision.action.value} ===\n")
        
        explanation.append("DISPATCH:")
        if decision.battery_kw != 0:
            direction = "Charging" if decision.battery_kw > 0 else "Discharging"
            explanation.append(f"  Battery: {direction} {abs(decision.battery_kw):.1f} kW")
        if decision.diesel_kw > 0:
            explanation.append(f"  Diesel: {decision.diesel_kw:.1f} kW")
        if decision.v2g_kw > 0:
            explanation.append(f"  V2G: {decision.v2g_kw:.1f} kW")
        if decision.grid_kw != 0:
            direction = "Import" if decision.grid_kw > 0 else "Export"
            explanation.append(f"  Grid: {direction} {abs(decision.grid_kw):.1f} kW")
        
        explanation.append("\nCONSTRAINT EVALUATION:")
        for priority in ConstraintPriority:
            nodes = [n for n in decision.decision_tree if n.priority == priority]
            if nodes:
                explanation.append(f"  {priority.name}:")
                for node in nodes:
                    status = "✓" if node.satisfied else "✗"
                    explanation.append(f"    {status} {node.message}")
        
        if decision.active_constraints:
            explanation.append("\nACTIVE CONSTRAINTS:")
            for constraint in decision.active_constraints:
                explanation.append(f"  • {constraint}")
        
        explanation.append(f"\nOBJECTIVE SCORE: ${decision.objective_score:.2f}")
        explanation.append(f"OPPORTUNITY COST: ${decision.opportunity_cost:.2f}")
        
        if decision.future_actions:
            explanation.append("\nNEXT 15-MIN PLAN:")
            for time, action in decision.future_actions:
                explanation.append(f"  t+{time:.0f}min: {action}")
        
        return "\n".join(explanation)
    
    def calculate_opportunity_cost(
        self,
        battery_kw: float,
        diesel_kw: float,
        grid_kw: float,
        prefs: UserPreferences
    ) -> float:
        """
        Calculate what alternative actions would have cost.
        Returns cost difference from optimal baseline.
        """
        # Current cost
        current_cost = 0.0
        current_cost += abs(battery_kw) * 0.02 * prefs.battery_wear_weight
        current_cost += diesel_kw * prefs.diesel_cost_per_kwh
        current_cost += max(0, grid_kw) * prefs.grid_import_cost_per_kwh
        
        # Alternative: all from grid
        total_kw = abs(battery_kw) + diesel_kw
        grid_alternative = total_kw * prefs.grid_import_cost_per_kwh
        
        return abs(grid_alternative - current_cost)
    
    def get_active_constraints(
        self,
        decision: Optional[RoutingDecision] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns list of currently binding constraints.
        """
        if decision is None:
            if not self.decision_history:
                return []
            decision = self.decision_history[-1]
        
        active = []
        for node in decision.decision_tree:
            if node.active and not node.satisfied:
                active.append({
                    'constraint': node.constraint,
                    'priority': node.priority.name,
                    'value': node.value,
                    'limit': node.limit,
                    'violation': abs(node.value - float(node.limit)) if isinstance(node.limit, (int, float)) else 0
                })
        
        return sorted(active, key=lambda x: ConstraintPriority[x['priority']].value
