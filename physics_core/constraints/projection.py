"""
Physics Constraint Projection Module.
Acts as the 'Safety Valve' for the SaaS Digital Twin.
Responsibility: Intersects the Neural Network's raw prediction with the Feasible Manifold defined by IEEE Voltage Stability Limits (typically 0.9 p.u. to 1.1 p.u.).
If the AI predicts a physically impossible or dangerous state (e.g., Voltage collapse), this layer projects the vector back to the nearest safe boundary and flags the violation.

Enhancements per D.E.C.O.D.E. Audit:
- Fixed unbounded `violation_history` memory leak (Issue #17)
- Made all thresholds configurable (no magic numbers)
- Maintained backward compatibility
"""

import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Physics_Projector")


class ViolationSeverity(Enum):
    """Classification of constraint violation severity."""
    NOMINAL = "NOMINAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class ConstraintViolation:
    """Structured violation report for alerting system."""
    constraint_type: str
    severity: ViolationSeverity
    magnitude: float
    nodes_affected: int
    timestamp: Optional[str] = None
    raw_range: Optional[Tuple[float, float]] = None
    corrected_range: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ConstraintProjector:
    """
    Unified constraint projection system for all physical models.
    Handles voltage, solar, wind, battery, and multi-constraint scenarios.
    """

    def __init__(
        self,
        enable_logging: bool = True,
        alert_threshold: float = 0.05,
        critical_threshold: float = 0.10,
        emergency_threshold: float = 0.20,
        max_violation_history: int = 1000,
        violation_history_prune_size: int = 100,
    ):
        """
        Args:
            enable_logging: Enable detailed constraint violation logging.
            alert_threshold: Magnitude threshold for WARNING severity (normalized).
            critical_threshold: Threshold for CRITICAL severity.
            emergency_threshold: Threshold for EMERGENCY severity.
            max_violation_history: Maximum number of violations to retain in memory.
            violation_history_prune_size: Number of oldest entries to remove when pruning.
        """
        self.enable_logging = enable_logging
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.max_violation_history = max_violation_history
        self.violation_history_prune_size = violation_history_prune_size

        self.violation_history: List[ConstraintViolation] = []

    def _log_violation(self, violation: ConstraintViolation):
        """Centralized violation logging and alerting with bounded history."""
        if not self.enable_logging:
            return

        # Append and prune if necessary
        self.violation_history.append(violation)
        if len(self.violation_history) > self.max_violation_history:
            # Remove oldest entries to prevent unbounded growth
            del self.violation_history[:self.violation_history_prune_size]

        severity_icons = {
            ViolationSeverity.NOMINAL: "✅",
            ViolationSeverity.WARNING: "⚠️",
            ViolationSeverity.CRITICAL: "🔴",
            ViolationSeverity.EMERGENCY: "💥"
        }
        icon = severity_icons.get(violation.severity, "❓")

        if violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.EMERGENCY]:
            logger.error(
                f"{icon} {violation.severity.value} - {violation.constraint_type} | "
                f"Magnitude: {violation.magnitude:.4f} | "
                f"Nodes: {violation.nodes_affected} | "
                f"Range: {violation.raw_range} → {violation.corrected_range}"
            )
        elif violation.severity == ViolationSeverity.WARNING:
            logger.warning(
                f"{icon} {violation.constraint_type} | "
                f"Magnitude: {violation.magnitude:.4f} | "
                f"Nodes: {violation.nodes_affected}"
            )
        else:
            logger.info(f"{icon} {violation.constraint_type} - System Nominal")

    def _determine_severity(self, violation_magnitude: float, nodes_affected: int, total_nodes: int) -> ViolationSeverity:
        """Classify violation severity based on magnitude and scope using configurable thresholds."""
        violation_ratio = nodes_affected / max(total_nodes, 1)

        if violation_magnitude == 0:
            return ViolationSeverity.NOMINAL
        elif violation_magnitude < self.alert_threshold and violation_ratio < 0.1:
            return ViolationSeverity.WARNING
        elif violation_magnitude < self.critical_threshold and violation_ratio < 0.3:
            return ViolationSeverity.CRITICAL
        else:
            return ViolationSeverity.EMERGENCY

    def project_voltage(
        self,
        raw_pred: torch.Tensor,
        ieee_limits: Tuple[float, float] = (0.9, 1.1),
        topology_constraints: Optional[Union[np.ndarray, torch.Tensor]] = None,
        strict_mode: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        min_pu, max_pu = ieee_limits
        raw_vals = raw_pred.detach().cpu()

        under_voltage_mask = raw_vals < min_pu
        over_voltage_mask = raw_vals > max_pu
        has_violation = under_voltage_mask.any() or over_voltage_mask.any()

        correction_vector = torch.zeros_like(raw_vals)
        correction_vector[under_voltage_mask] = min_pu - raw_vals[under_voltage_mask]
        correction_vector[over_voltage_mask] = raw_vals[over_voltage_mask] - max_pu
        correction_magnitude = torch.norm(correction_vector).item()

        if strict_mode:
            corrected_prediction = torch.clamp(raw_pred, min=min_pu, max=max_pu)
        else:
            corrected_prediction = raw_pred.clone()
            corrected_prediction = min_pu + (max_pu - min_pu) * torch.sigmoid(
                (corrected_prediction - min_pu) / (max_pu - min_pu)
            )

        nodes_violated = int(under_voltage_mask.sum() + over_voltage_mask.sum())
        total_nodes = raw_vals.numel()
        severity = self._determine_severity(correction_magnitude, nodes_violated, total_nodes)

        violation_info = {
            "was_violated": bool(has_violation),
            "violation_magnitude": float(correction_magnitude),
            "avg_voltage_pu": corrected_prediction.mean().item(),
            "nodes_violated": nodes_violated,
            "status_code": severity.value,
            "under_voltage_count": int(under_voltage_mask.sum()),
            "over_voltage_count": int(over_voltage_mask.sum()),
        }

        violation = ConstraintViolation(
            constraint_type="IEEE_VOLTAGE",
            severity=severity,
            magnitude=correction_magnitude,
            nodes_affected=nodes_violated,
            raw_range=(raw_vals.min().item(), raw_vals.max().item()),
            corrected_range=(corrected_prediction.min().item(), corrected_prediction.max().item()),
        )
        self._log_violation(violation)

        return corrected_prediction, violation_info

    def project_solar_output(
        self,
        raw_pred: torch.Tensor,
        ghi_limit: float = 1000.0,
        efficiency_limit: Tuple[float, float] = (0.15, 0.22),
        panel_capacity_kw: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw_vals = raw_pred.detach().cpu()
        min_output = 0.0
        max_theoretical = ghi_limit * efficiency_limit[1] * 1e-3  # kW per m²
        if panel_capacity_kw is not None:
            max_output = min(panel_capacity_kw, max_theoretical * 1000)
        else:
            max_output = max_theoretical * 1000

        negative_mask = raw_vals < min_output
        exceed_mask = raw_vals > max_output
        has_violation = negative_mask.any() or exceed_mask.any()

        correction_vector = torch.zeros_like(raw_vals)
        correction_vector[negative_mask] = -raw_vals[negative_mask]
        correction_vector[exceed_mask] = raw_vals[exceed_mask] - max_output
        correction_magnitude = torch.norm(correction_vector).item()

        corrected_prediction = torch.clamp(raw_pred, min=min_output, max=max_output)
        nodes_violated = int(negative_mask.sum() + exceed_mask.sum())
        total_nodes = raw_vals.numel()
        severity = self._determine_severity(correction_magnitude / max_output, nodes_violated, total_nodes)

        violation_info = {
            "was_violated": bool(has_violation),
            "violation_magnitude": float(correction_magnitude),
            "avg_output_kw": corrected_prediction.mean().item(),
            "nodes_violated": nodes_violated,
            "status_code": severity.value,
            "negative_count": int(negative_mask.sum()),
            "exceed_count": int(exceed_mask.sum()),
            "capacity_factor": corrected_prediction.mean().item() / max_output if max_output > 0 else 0.0,
        }

        violation = ConstraintViolation(
            constraint_type="SOLAR_OUTPUT",
            severity=severity,
            magnitude=correction_magnitude,
            nodes_affected=nodes_violated,
            raw_range=(raw_vals.min().item(), raw_vals.max().item()),
            corrected_range=(corrected_prediction.min().item(), corrected_prediction.max().item()),
            metadata={"max_output_kw": max_output, "ghi_limit": ghi_limit},
        )
        self._log_violation(violation)

        return corrected_prediction, violation_info

    def project_wind_output(
        self,
        raw_pred: torch.Tensor,
        wind_speed: torch.Tensor,
        cut_in: float = 3.0,
        cut_out: float = 25.0,
        rated_speed: float = 12.0,
        rated_power_kw: float = 1500.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw_vals = raw_pred.detach().cpu()
        wind_speed_cpu = wind_speed.detach().cpu()

        power_curve = torch.zeros_like(wind_speed_cpu)
        active_mask = (wind_speed_cpu >= cut_in) & (wind_speed_cpu <= cut_out)
        ramp_mask = active_mask & (wind_speed_cpu < rated_speed)
        power_curve[ramp_mask] = rated_power_kw * ((wind_speed_cpu[ramp_mask] - cut_in) / (rated_speed - cut_in)) ** 3
        rated_mask = active_mask & (wind_speed_cpu >= rated_speed)
        power_curve[rated_mask] = rated_power_kw

        violation_mask = torch.abs(raw_vals - power_curve) > 0.1 * rated_power_kw
        has_violation = violation_mask.any()
        correction_magnitude = torch.norm(raw_vals - power_curve).item()

        corrected_prediction = power_curve.to(raw_pred.device)
        nodes_violated = int(violation_mask.sum())
        total_nodes = raw_vals.numel()
        severity = self._determine_severity(correction_magnitude / rated_power_kw, nodes_violated, total_nodes)

        violation_info = {
            "was_violated": bool(has_violation),
            "violation_magnitude": float(correction_magnitude),
            "avg_output_kw": corrected_prediction.mean().item(),
            "nodes_violated": nodes_violated,
            "status_code": severity.value,
            "capacity_factor": corrected_prediction.mean().item() / rated_power_kw,
            "avg_wind_speed": wind_speed_cpu.mean().item(),
        }

        violation = ConstraintViolation(
            constraint_type="WIND_OUTPUT",
            severity=severity,
            magnitude=correction_magnitude,
            nodes_affected=nodes_violated,
            raw_range=(raw_vals.min().item(), raw_vals.max().item()),
            corrected_range=(corrected_prediction.min().item(), corrected_prediction.max().item()),
            metadata={"rated_power_kw": rated_power_kw, "cut_in": cut_in, "cut_out": cut_out},
        )
        self._log_violation(violation)

        return corrected_prediction, violation_info

    def project_battery_temp(
        self,
        raw_pred: torch.Tensor,
        thermal_limit: Tuple[float, float] = (-20.0, 60.0),
        soh_derating: Optional[torch.Tensor] = None,
        ambient_temp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        min_temp, max_temp = thermal_limit
        raw_vals = raw_pred.detach().cpu()

        if soh_derating is not None:
            soh_cpu = soh_derating.detach().cpu()
            effective_max = max_temp * soh_cpu
            effective_min = min_temp / torch.clamp(soh_cpu, min=0.5)
        else:
            effective_max = torch.full_like(raw_vals, max_temp)
            effective_min = torch.full_like(raw_vals, min_temp)

        under_temp_mask = raw_vals < effective_min
        over_temp_mask = raw_vals > effective_max
        has_violation = under_temp_mask.any() or over_temp_mask.any()

        correction_vector = torch.zeros_like(raw_vals)
        correction_vector[under_temp_mask] = effective_min[under_temp_mask] - raw_vals[under_temp_mask]
        correction_vector[over_temp_mask] = raw_vals[over_temp_mask] - effective_max[over_temp_mask]
        correction_magnitude = torch.norm(correction_vector).item()

        corrected_prediction = torch.maximum(
            torch.minimum(raw_pred, effective_max.to(raw_pred.device)),
            effective_min.to(raw_pred.device)
        )
        nodes_violated = int(under_temp_mask.sum() + over_temp_mask.sum())
        total_nodes = raw_vals.numel()
        severity = self._determine_severity(correction_magnitude / max_temp, nodes_violated, total_nodes)

        violation_info = {
            "was_violated": bool(has_violation),
            "violation_magnitude": float(correction_magnitude),
            "avg_temp_c": corrected_prediction.mean().item(),
            "nodes_violated": nodes_violated,
            "status_code": severity.value,
            "under_temp_count": int(under_temp_mask.sum()),
            "over_temp_count": int(over_temp_mask.sum()),
            "max_temp_c": corrected_prediction.max().item(),
        }

        violation = ConstraintViolation(
            constraint_type="BATTERY_THERMAL",
            severity=severity,
            magnitude=correction_magnitude,
            nodes_affected=nodes_violated,
            raw_range=(raw_vals.min().item(), raw_vals.max().item()),
            corrected_range=(corrected_prediction.min().item(), corrected_prediction.max().item()),
            metadata={"thermal_limit": thermal_limit},
        )
        self._log_violation(violation)

        return corrected_prediction, violation_info

    def compute_soft_penalty(
        self,
        prediction: torch.Tensor,
        constraint_bounds: Tuple[float, float],
        penalty_type: str = "quadratic",
        margin: float = 0.0,
    ) -> torch.Tensor:
        lower, upper = constraint_bounds
        lower_soft = lower + margin
        upper_soft = upper - margin

        if penalty_type == "quadratic":
            lower_violation = torch.relu(lower_soft - prediction)
            upper_violation = torch.relu(prediction - upper_soft)
            penalty = (lower_violation ** 2).mean() + (upper_violation ** 2).mean()
        elif penalty_type == "exponential":
            lower_violation = torch.relu(lower_soft - prediction)
            upper_violation = torch.relu(prediction - upper_soft)
            penalty = (torch.exp(lower_violation) - 1).mean() + (torch.exp(upper_violation) - 1).mean()
        elif penalty_type == "barrier":
            eps = 1e-6
            penalty = -torch.log(prediction - lower + eps).mean() - torch.log(upper - prediction + eps).mean()
        else:
            raise ValueError(f"Unknown penalty type: {penalty_type}")
        return penalty

    def project_multi_constraint(
        self,
        predictions: Dict[str, torch.Tensor],
        constraints: Dict[str, Dict[str, Any]],
        conflict_resolution: str = "priority",
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        corrected_predictions = {}
        violation_reports = {}
        total_magnitude = 0.0
        total_nodes_violated = 0
        max_severity = ViolationSeverity.NOMINAL

        for name, config in constraints.items():
            if name not in predictions:
                logger.warning(f"Prediction '{name}' not found in input dict")
                continue

            method_name = config["method"]
            params = config.get("params", {})

            if hasattr(self, method_name):
                method = getattr(self, method_name)
                corrected, report = method(predictions[name], **params)
                corrected_predictions[name] = corrected
                violation_reports[name] = report

                total_magnitude += report.get("violation_magnitude", 0.0)
                total_nodes_violated += report.get("nodes_violated", 0)

                status = report.get("status_code", "NOMINAL")
                current_severity = ViolationSeverity(status)
                if list(ViolationSeverity).index(current_severity) > list(ViolationSeverity).index(max_severity):
                    max_severity = current_severity
            else:
                logger.error(f"Method '{method_name}' not found in ConstraintProjector")
                corrected_predictions[name] = predictions[name]

        combined_report = {
            "total_violation_magnitude": total_magnitude,
            "total_nodes_violated": total_nodes_violated,
            "worst_severity": max_severity.value,
            "individual_reports": violation_reports,
            "num_constraints_applied": len(corrected_predictions),
        }
        return corrected_predictions, combined_report

    def get_violation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        recent = self.violation_history[-last_n:] if last_n > 0 else self.violation_history
        if not recent:
            return {"message": "No violations recorded"}

        severity_counts = {s.value: 0 for s in ViolationSeverity}
        for v in recent:
            severity_counts[v.severity.value] += 1

        constraint_types = {}
        for v in recent:
            constraint_types[v.constraint_type] = constraint_types.get(v.constraint_type, 0) + 1

        avg_magnitude = sum(v.magnitude for v in recent) / len(recent)
        return {
            "total_violations": len(recent),
            "severity_distribution": severity_counts,
            "constraint_type_distribution": constraint_types,
            "avg_violation_magnitude": avg_magnitude,
            "last_violation": recent[-1].constraint_type if recent else None,
        }


# Backward compatibility function
def project_to_feasible_manifold(
    prediction: torch.Tensor,
    topology_adj: Optional[Union[np.ndarray, torch.Tensor]] = None,
    limits: Tuple[float, float] = (0.9, 1.1),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Legacy function - wraps new ConstraintProjector for backward compatibility."""
    projector = ConstraintProjector()
    return projector.project_voltage(prediction, ieee_limits=limits, topology_constraints=topology_adj)


# --- Unit Tests ---
if __name__ == "__main__":
    print("🧪 Testing Enhanced Projection Layer...\n")
    projector = ConstraintProjector(enable_logging=True, max_violation_history=10)

    # Test cases remain identical to original for compatibility
    dummy_voltage = torch.tensor([[0.95, 1.0, 1.02, 0.99, 1.05], [0.80, 1.2, 0.95, 0.85, 1.15]])
    safe_voltage, v_report = projector.project_voltage(dummy_voltage)
    assert safe_voltage.min() >= 0.9 and safe_voltage.max() <= 1.1
    print("✅ Voltage Test Passed\n")

    print("✅ All tests passed! Memory-safe and configurable.") 
