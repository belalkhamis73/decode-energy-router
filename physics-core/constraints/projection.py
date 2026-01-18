"""
Physics Constraint Projection Module.
Acts as the 'Safety Valve' for the SaaS Digital Twin.

Responsibility:
Intersects the Neural Network's raw prediction with the Feasible Manifold 
defined by IEEE Voltage Stability Limits (typically 0.9 p.u. to 1.1 p.u.).

If the AI predicts a physically impossible or dangerous state (e.g., Voltage collapse),
this layer projects the vector back to the nearest safe boundary and flags the violation.
"""

import torch
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Union

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Physics_Projector")

def project_to_feasible_manifold(
    prediction: torch.Tensor, 
    topology_adj: Optional[Union[np.ndarray, torch.Tensor]] = None,
    limits: Tuple[float, float] = (0.9, 1.1)
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Projects raw DeepONet voltage predictions onto the operational limits of the grid.
    
    Logic:
    1. Checks if any bus voltage V_i violates V_min <= V_i <= V_max.
    2. Calculates the magnitude of the violation (Stability Index).
    3. Clamps (projects) violating values to the nearest boundary.
    
    Args:
        prediction: Raw output tensor from DeepONet [Batch, N_Buses] (Voltage p.u.)
        topology_adj: Adjacency matrix of the grid. 
                      (Reserved for future Kirchhoff's Current Law checks).
        limits: Tuple of (Min_Voltage, Max_Voltage). Default IEEE (0.9, 1.1).
        
    Returns:
        corrected_prediction: Tensor guaranteed to be within safe limits.
        violation_info: Dictionary containing safety metrics for the SaaS Dashboard.
    """
    min_pu, max_pu = limits
    
    # Ensure we are working with a detached tensor for safety checks
    # (We don't want gradients flowing through the safety report logic)
    raw_vals = prediction.detach().cpu()
    
    # 1. Detect Violations
    # Create boolean masks for under-voltage and over-voltage
    under_voltage_mask = raw_vals < min_pu
    over_voltage_mask = raw_vals > max_pu
    has_violation = under_voltage_mask.any() or over_voltage_mask.any()
    
    # 2. Calculate Violation Magnitude (L2 Norm of the error vector)
    # How far off was the AI?
    correction_vector = torch.zeros_like(raw_vals)
    correction_vector[under_voltage_mask] = min_pu - raw_vals[under_voltage_mask]
    correction_vector[over_voltage_mask] = max_pu - raw_vals[over_voltage_mask]
    
    correction_magnitude = torch.norm(correction_vector).item()
    
    # 3. Apply Projection (Clamping)
    # This forces the output to sit exactly on the boundary of the feasible set
    corrected_prediction = torch.clamp(prediction, min=min_pu, max=max_pu)
    
    # 4. Global Stability Metrics
    # Average Voltage helps determine if the grid is "sagging" (overall heavy load)
    avg_voltage = corrected_prediction.mean().item()
    
    # Construct Safety Report
    violation_info = {
        "was_violated": bool(has_violation),
        "violation_magnitude": float(correction_magnitude),
        "avg_voltage_pu": avg_voltage,
        "nodes_violated": int(under_voltage_mask.sum() + over_voltage_mask.sum()),
        "status_code": "CRITICAL" if has_violation else "NOMINAL"
    }
    
    if has_violation:
        logger.warning(
            f"⚠️ Physics Violation Detected! "
            f"Magnitude: {correction_magnitude:.4f} | "
            f"Nodes Affected: {violation_info['nodes_violated']}"
        )
        logger.warning(f"   -> Projection applied. Output clamped to [{min_pu}, {max_pu}] p.u.")
        
    return corrected_prediction, violation_info

# --- Unit Test (Manual Verification) ---
if __name__ == "__main__":
    print("🛡️  Testing Projection Layer (Safety Valve)...")
    
    # Simulate a batch of 2 samples, 5 buses each
    # Sample 0: Safe [0.95 ... 1.05]
    # Sample 1: Unsafe [0.8 (Under), 1.2 (Over), ... ]
    dummy_pred = torch.tensor([
        [0.95, 1.0, 1.02, 0.99, 1.05],
        [0.80, 1.2, 0.95, 0.85, 1.15]
    ])
    
    print(f"   Input Range: [{dummy_pred.min():.2f}, {dummy_pred.max():.2f}]")
    
    safe_tensor, report = project_to_feasible_manifold(dummy_pred, limits=(0.9, 1.1))
    
    print(f"   Output Range: [{safe_tensor.min():.2f}, {safe_tensor.max():.2f}]")
    print(f"   Safety Report: {report}")
    
    assert safe_tensor.min() >= 0.9
    assert safe_tensor.max() <= 1.1
    assert report["was_violated"] is True
    print("✅ Test Passed: Constraints Enforced.")
  
