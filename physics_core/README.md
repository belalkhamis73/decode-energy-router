# ⚛️ Physics Core

**The immutable laws of nature that govern the Digital Twin.**

Unlike standard AI models that "guess" patterns, this module enforces hard physical constraints using PyTorch. These equations differ from standard simulation tools (like MATLAB/Simulink) because they are **differentiable**, allowing gradients to flow back through the laws of physics to update the Neural Network.

## Included Equations

| Equation | Domain | Implementation |
| :--- | :--- | :--- |
| **Swing Equation** | Frequency Stability (Inertia) | `swing_equation.py` |
| **Power Flow (AC)** | Grid Equilibrium (Kirchhoff) | `power_flow.py` |
| **Lumped Thermal** | Battery Safety (Thermodynamics) | `battery_thermal.py` |

## Usage Principle

1. **Forward Mode:** Calculate the "Residual" (Error). If `Residual == 0`, physics is satisfied.
2. **Backward Mode:** Use the Residual as a Loss Function (`Loss_Physics`) to train the PINN.

## Constants
All grid parameters (Line Impedance, Inertia Constants) are centralized in `constants/grid_parameters.json` to ensure the Simulation and the AI Model utilize the exact same physical topology.
