"""
Unified Physics Validation Engine
Combines all governing equations into a single differentiable loss function
for comprehensive physics compliance checking.

Integrates:
1. Power Flow Equations (KCL/KVL)
2. Swing Equation (Frequency Dynamics)
3. Battery Thermal Dynamics
4. Battery Health Degradation
5. Energy Conservation Laws
6. Constraint Manifolds

Provides:
- Real-time physics violation detection
- Composite residual calculation
- Severity classification
- Monitoring system integration
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum
import logging
from datetime import datetime

# Import physics equations
from .power_flow import PowerFlowEquation
from .swing_equation import SwingEquation, calculate_freq_deviation
from .battery_thermal import BatteryThermalEquation
from .battery_health import BatteryHealthEquation
from .energy_router import SourceState

logger = logging.getLogger("UnifiedPhysics")


class ViolationSeverity(Enum):
    """Physics violation severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of physics violations."""
    POWER_BALANCE = "power_balance"
    VOLTAGE_LIMITS = "voltage_limits"
    FREQUENCY_DEVIATION = "frequency_deviation"
    THERMAL_LIMIT = "thermal_limit"
    SOC_LIMIT = "soc_limit"
    ENERGY_CONSERVATION = "energy_conservation"
    SWING_EQUATION = "swing_equation"
    POWER_FLOW = "power_flow"


class PhysicsViolation:
    """Represents a single physics violation."""
    
    def __init__(
        self,
        violation_type: ViolationType,
        severity: ViolationSeverity,
        residual: float,
        threshold: float,
        description: str,
        timestamp: Optional[datetime] = None
    ):
        self.violation_type = violation_type
        self.severity = severity
        self.residual = residual
        self.threshold = threshold
        self.description = description
        self.timestamp = timestamp or datetime.utcnow()
        
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/monitoring."""
        return {
            "type": self.violation_type.value,
            "severity": self.severity.value,
            "residual": float(self.residual),
            "threshold": float(self.threshold),
            "description": self.description,
            "timestamp": self.timestamp.isoformat()
        }


class UnifiedPhysicsValidator(nn.Module):
    """
    The "Ground Truth" Engine for the Digital Twin.
    Aggregates all physics residuals into a composite loss.
    """
    
    def __init__(self, grid_topology: Dict):
        super().__init__()
        
        # Initialize sub-modules
        self.power_flow = PowerFlowEquation(grid_topology['n_buses'])
        self.swing_eq = SwingEquation(
            inertia_h=grid_topology.get('inertia_h', 3.5),
            damping_d=grid_topology.get('damping_d', 1.0)
        )
        self.thermal = BatteryThermalEquation(
            mass_kg=grid_topology.get('battery_mass_kg', 250.0),
            specific_heat_cp=900.0,
            internal_resistance_r=grid_topology.get('battery_resistance', 0.05),
            heat_transfer_coeff_h=15.0,
            surface_area_a=2.5
        )
        self.health = BatteryHealthEquation()
        
        # Loss weights (configurable via config.py)
        self.register_buffer('w_power', torch.tensor(1.0))
        self.register_buffer('w_swing', torch.tensor(0.5))
        self.register_buffer('w_thermal', torch.tensor(0.3))
        self.register_buffer('w_health', torch.tensor(0.2))
        self.register_buffer('w_energy', torch.tensor(0.8))
        
        # Physics compliance thresholds
        self.thresholds = {
            'power_balance': 1e-3,      # 0.1% power imbalance
            'swing_violation': 1e-4,     # Frequency/angle deviation
            'thermal_violation': 1.0,    # 1°C temperature deviation
            'voltage_violation': 0.05,   # 5% voltage deviation
            'soc_violation': 0.02,       # 2% SOC deviation
            'energy_conservation': 1e-2, # 1% energy loss/gain
            'composite_loss': 1e-4       # Overall physics compliance
        }
        
        # Severity thresholds (multiples of base threshold)
        self.severity_multipliers = {
            ViolationSeverity.LOW: 1.0,
            ViolationSeverity.MEDIUM: 2.0,
            ViolationSeverity.HIGH: 5.0,
            ViolationSeverity.CRITICAL: 10.0
        }
        
        # Violation history for monitoring
        self.violation_history: List[PhysicsViolation] = []
        self.max_history_size = 1000
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        measurements: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate physics residuals across all domains.
        
        Args:
            predictions: Model predictions (V_mag, V_ang, omega, etc.)
            measurements: Ground truth measurements (P_injection, Q_injection, etc.)
            
        Returns:
            Dictionary containing:
                - residuals: Individual physics violations
                - composite_loss: Weighted sum for backpropagation
                - is_valid: Boolean physics compliance flag
        """
        residuals = {}
        
        # 1. Power Flow Residuals (KCL/KVL)
        res_P, res_Q = self.power_flow(
            predictions['V_mag'], 
            predictions['V_ang'],
            measurements['P_injection'], 
            measurements['Q_injection']
        )
        residuals['power_balance'] = torch.norm(res_P) + torch.norm(res_Q)
        
        # 2. Swing Equation Residuals (Frequency Dynamics)
        res_delta, res_omega = self.swing_eq(
            predictions['omega'], 
            predictions['d_delta_dt'], 
            predictions['d_omega_dt'],
            measurements['P_mech'], 
            measurements['P_elec']
        )
        residuals['swing_violation'] = torch.norm(res_delta) + torch.norm(res_omega)
        
        # 3. Thermal Residuals (Battery Safety)
        if 'battery_temp' in predictions:
            res_thermal = self.thermal(
                predictions['battery_temp'], 
                predictions['d_temp_dt'],
                measurements['battery_current'], 
                measurements['temp_ambient']
            )
            residuals['thermal_violation'] = torch.norm(res_thermal)
        
        # 4. Voltage Limit Violations
        if 'V_mag' in predictions:
            v_min, v_max = 0.95, 1.05
            voltage_violation = torch.clamp(predictions['V_mag'] - v_max, min=0) + \
                               torch.clamp(v_min - predictions['V_mag'], min=0)
            residuals['voltage_violation'] = torch.sum(voltage_violation)
        
        # 5. Energy Conservation (power balance over time)
        if 'energy_in' in measurements and 'energy_out' in measurements:
            energy_imbalance = torch.abs(
                measurements['energy_in'] - measurements['energy_out']
            )
            total_energy = measurements['energy_in'] + 1e-6
            residuals['energy_conservation'] = energy_imbalance / total_energy
        
        # Composite Loss
        composite = (
            self.w_power * residuals.get('power_balance', torch.tensor(0.0)) +
            self.w_swing * residuals.get('swing_violation', torch.tensor(0.0))
        )
        
        if 'thermal_violation' in residuals:
            composite += self.w_thermal * residuals['thermal_violation']
            
        if 'voltage_violation' in residuals:
            composite += residuals['voltage_violation']
            
        if 'energy_conservation' in residuals:
            composite += self.w_energy * residuals['energy_conservation']
        
        is_valid = composite < self.thresholds['composite_loss']
        
        return {
            'residuals': residuals,
            'composite_loss': composite,
            'is_valid': is_valid
        }
    
    def validate_all_constraints(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check all physics laws simultaneously against current state.
        
        Args:
            state: Current system state with all variables
            
        Returns:
            Dictionary containing:
                - valid: Overall validity flag
                - violations: List of PhysicsViolation objects
                - residuals: Raw residual values
                - severity: Overall severity level
        """
        violations = []
        residuals = {}
        
        # Convert state to tensors
        with torch.no_grad():
            # 1. Power Balance Check
            total_generation = state.get('solar_kw', 0) + \
                             state.get('wind_kw', 0) + \
                             state.get('battery_kw', 0) + \
                             state.get('diesel_kw', 0)
            total_load = state.get('load_kw', 0)
            power_imbalance = abs(total_generation - total_load)
            power_imbalance_pu = power_imbalance / (total_load + 1e-6)
            
            residuals['power_balance'] = power_imbalance_pu
            
            if power_imbalance_pu > self.thresholds['power_balance']:
                severity = self._classify_severity(
                    power_imbalance_pu,
                    self.thresholds['power_balance']
                )
                violations.append(PhysicsViolation(
                    violation_type=ViolationType.POWER_BALANCE,
                    severity=severity,
                    residual=power_imbalance_pu,
                    threshold=self.thresholds['power_balance'],
                    description=f"Power imbalance: {power_imbalance:.2f} kW "
                               f"({power_imbalance_pu*100:.2f}%)"
                ))
            
            # 2. Voltage Limit Check
            voltage_pu = state.get('voltage_pu', 1.0)
            voltage_deviation = max(
                abs(voltage_pu - 1.0) - 0.05,  # ±5% tolerance
                0.0
            )
            
            residuals['voltage_violation'] = voltage_deviation
            
            if voltage_deviation > 0:
                severity = self._classify_severity(
                    voltage_deviation,
                    self.thresholds['voltage_violation']
                )
                violations.append(PhysicsViolation(
                    violation_type=ViolationType.VOLTAGE_LIMITS,
                    severity=severity,
                    residual=voltage_deviation,
                    threshold=self.thresholds['voltage_violation'],
                    description=f"Voltage out of limits: {voltage_pu:.4f} pu"
                ))
            
            # 3. Frequency Deviation Check
            freq_hz = state.get('frequency_hz', 60.0)
            freq_deviation = abs(freq_hz - 60.0)
            
            residuals['frequency_deviation'] = freq_deviation
            
            if freq_deviation > 0.1:  # ±0.1 Hz tolerance
                severity = self._classify_severity(freq_deviation, 0.1)
                violations.append(PhysicsViolation(
                    violation_type=ViolationType.FREQUENCY_DEVIATION,
                    severity=severity,
                    residual=freq_deviation,
                    threshold=0.1,
                    description=f"Frequency deviation: {freq_deviation:.3f} Hz"
                ))
            
            # 4. Thermal Limit Check
            battery_temp = state.get('battery_temp_c', 25.0)
            temp_violation = max(battery_temp - 50.0, 0.0)  # Max 50°C
            
            residuals['thermal_violation'] = temp_violation
            
            if temp_violation > 0:
                severity = self._classify_severity(
                    temp_violation,
                    self.thresholds['thermal_violation']
                )
                violations.append(PhysicsViolation(
                    violation_type=ViolationType.THERMAL_LIMIT,
                    severity=severity,
                    residual=temp_violation,
                    threshold=50.0,
                    description=f"Battery overtemperature: {battery_temp:.1f}°C"
                ))
            
            # 5. SOC Limit Check
            battery_soc = state.get('battery_soc', 0.5)
            soc_violation = max(0.1 - battery_soc, 0.0) + \
                           max(battery_soc - 0.95, 0.0)
            
            residuals['soc_violation'] = soc_violation
            
            if soc_violation > 0:
                severity = self._classify_severity(
                    soc_violation,
                    self.thresholds['soc_violation']
                )
                violations.append(PhysicsViolation(
                    violation_type=ViolationType.SOC_LIMIT,
                    severity=severity,
                    residual=soc_violation,
                    threshold=self.thresholds['soc_violation'],
                    description=f"SOC out of safe range: {battery_soc*100:.1f}%"
                ))
        
        # Determine overall severity
        overall_severity = ViolationSeverity.NONE
        if violations:
            severity_order = [
                ViolationSeverity.NONE,
                ViolationSeverity.LOW,
                ViolationSeverity.MEDIUM,
                ViolationSeverity.HIGH,
                ViolationSeverity.CRITICAL
            ]
            overall_severity = max(
                (v.severity for v in violations),
                key=lambda s: severity_order.index(s)
            )
        
        # Add to history
        self._update_violation_history(violations)
        
        return {
            'valid': len(violations) == 0,
            'violations': [v.to_dict() for v in violations],
            'residuals': {k: float(v) for k, v in residuals.items()},
            'severity': overall_severity.value,
            'violation_count': len(violations)
        }
    
    def calculate_composite_residual(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Aggregate all model residuals into metrics.
        
        Args:
            predictions: Dictionary of model predictions with residuals
            
        Returns:
            Dictionary with aggregated residual metrics
        """
        residuals = {}
        
        # Extract individual model residuals
        for model_name, pred in predictions.items():
            if isinstance(pred, dict) and 'physics_residual' in pred:
                residuals[f"{model_name}_residual"] = float(pred['physics_residual'])
        
        # Calculate statistics
        if residuals:
            residual_values = list(residuals.values())
            composite_metrics = {
                'mean_residual': sum(residual_values) / len(residual_values),
                'max_residual': max(residual_values),
                'min_residual': min(residual_values),
                'residual_std': torch.std(torch.tensor(residual_values)).item(),
                'residuals_by_model': residuals
            }
        else:
            composite_metrics = {
                'mean_residual': 0.0,
                'max_residual': 0.0,
                'min_residual': 0.0,
                'residual_std': 0.0,
                'residuals_by_model': {}
            }
        
        # Add compliance flag
        composite_metrics['physics_compliant'] = \
            composite_metrics['max_residual'] < self.thresholds['composite_loss']
        
        return composite_metrics
    
    def get_violation_severity(
        self,
        residuals: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Classify violation criticality for each residual.
        
        Args:
            residuals: Dictionary of residual values
            
        Returns:
            Dictionary with severity classifications
        """
        severity_report = {
            'classifications': {},
            'overall_severity': ViolationSeverity.NONE.value,
            'critical_violations': [],
            'warnings': []
        }
        
        severity_order = [
            ViolationSeverity.NONE,
            ViolationSeverity.LOW,
            ViolationSeverity.MEDIUM,
            ViolationSeverity.HIGH,
            ViolationSeverity.CRITICAL
        ]
        max_severity = ViolationSeverity.NONE
        
        for residual_name, residual_value in residuals.items():
            # Get appropriate threshold
            threshold = self.thresholds.get(residual_name, 1e-3)
            
            # Classify severity
            severity = self._classify_severity(residual_value, threshold)
            severity_report['classifications'][residual_name] = {
                'value': residual_value,
                'threshold': threshold,
                'severity': severity.value,
                'ratio': residual_value / threshold if threshold > 0 else 0
            }
            
            # Track max severity
            if severity_order.index(severity) > severity_order.index(max_severity):
                max_severity = severity
            
            # Collect critical violations
            if severity == ViolationSeverity.CRITICAL:
                severity_report['critical_violations'].append({
                    'name': residual_name,
                    'value': residual_value,
                    'threshold': threshold
                })
            elif severity in [ViolationSeverity.HIGH, ViolationSeverity.MEDIUM]:
                severity_report['warnings'].append({
                    'name': residual_name,
                    'value': residual_value,
                    'severity': severity.value
                })
        
        severity_report['overall_severity'] = max_severity.value
        
        return severity_report
    
    def _classify_severity(
        self,
        residual: float,
        threshold: float
    ) -> ViolationSeverity:
        """Classify violation severity based on threshold multiples."""
        if residual <= threshold:
            return ViolationSeverity.NONE
        
        ratio = residual / threshold
        
        if ratio < self.severity_multipliers[ViolationSeverity.MEDIUM]:
            return ViolationSeverity.LOW
        elif ratio < self.severity_multipliers[ViolationSeverity.HIGH]:
            return ViolationSeverity.MEDIUM
        elif ratio < self.severity_multipliers[ViolationSeverity.CRITICAL]:
            return ViolationSeverity.HIGH
        else:
            return ViolationSeverity.CRITICAL
    
    def _update_violation_history(self, violations: List[PhysicsViolation]):
        """Maintain violation history with size limit."""
        self.violation_history.extend(violations)
        
        # Trim history if too large
        if len(self.violation_history) > self.max_history_size:
            self.violation_history = \
                self.violation_history[-self.max_history_size:]
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get statistics from violation history."""
        if not self.violation_history:
            return {
                'total_violations': 0,
                'by_type': {},
                'by_severity': {},
                'recent_critical': []
            }
        
        stats = {
            'total_violations': len(self.violation_history),
            'by_type': {},
            'by_severity': {},
            'recent_critical': []
        }
        
        # Count by type
        for v in self.violation_history:
            type_name = v.violation_type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1
            
            severity_name = v.severity.value
            stats['by_severity'][severity_name] = \
                stats['by_severity'].get(severity_name, 0) + 1
            
            # Collect recent critical violations
            if v.severity == ViolationSeverity.CRITICAL:
                stats['recent_critical'].append(v.to_dict())
        
        # Keep only 10 most recent critical violations
        stats['recent_critical'] = stats['recent_critical'][-10:]
        
        return stats
    
    def reset_violation_history(self):
        """Clear violation history."""
        self.violation_history.clear()
        logger.info("Violation history cleared")
    
    def publish_violations_to_monitoring(
        self,
        violations: List[Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Format violations for monitoring system integration.
        
        Args:
            violations: List of violation dictionaries
            session_id: Session identifier
            
        Returns:
            Formatted monitoring payload
        """
        monitoring_payload = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'violation_count': len(violations),
            'violations': violations,
            'requires_action': any(
                v.get('severity') in ['high', 'critical']
                for v in violations
            )
        }
        
        # Add recommended actions for critical violations
        critical_violations = [
            v for v in violations
            if v.get('severity') == 'critical'
        ]
        
        if critical_violations:
            monitoring_payload['recommended_actions'] = \
                self._get_recommended_actions(critical_violations)
        
        return monitoring_payload
    
    def _get_recommended_actions(
        self,
        critical_violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommended actions for critical violations."""
        actions = []
        
        for violation in critical_violations:
            v_type = violation.get('type')
            
            if v_type == ViolationType.POWER_BALANCE.value:
                actions.append("Adjust generation or activate reserve capacity")
            elif v_type == ViolationType.VOLTAGE_LIMITS.value:
                actions.append("Adjust reactive power compensation")
            elif v_type == ViolationType.THERMAL_LIMIT.value:
                actions.append("Reduce battery current or activate cooling")
            elif v_type == ViolationType.SOC_LIMIT.value:
                actions.append("Adjust battery dispatch strategy")
            elif v_type == ViolationType.FREQUENCY_DEVIATION.value:
                actions.append("Increase system inertia or governor response")
        
        return list(set(actions))  # Remove 
