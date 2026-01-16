"""
Physics Core Unit Tests.
Verifies that the Digital Twin's kernel adheres to the immutable laws of 
thermodynamics and electromagnetism.

Principles:
- Test-Driven Development (TDD): Defines expected physical behavior before model training.
- First Principles: Validates against analytical solutions (Swing Equation, Kirchhoff's Laws).
- Fail Fast: Any deviation from physics > tolerance causes immediate pipeline failure.
"""

import pytest
import torch
import numpy as np

# Import the "Ground Truth" equations
from physics_core.equations.swing_equation import SwingEquation
from physics_core.equations.power_flow import PowerFlowEquation
from physics_core.equations.battery_thermal import BatteryThermalEquation

class TestPhysicsLaws:
    
    def test_swing_equation_equilibrium(self):
        """
        Law: Newton's Second Law for Rotation (Grid Stability).
        Scenario: Synchronous generator at steady state.
        Expectation: If Mechanical Power == Electrical Power, Acceleration must be ZERO.
        """
        # 1. Setup Physical Constants
        H_inertia = 3.0  # seconds
        D_damping = 1.0  # p.u.
        swing = SwingEquation(inertia_h=H_inertia, damping_d=D_damping)

        # 2. Define Equilibrium State
        # No speed deviation (omega = 0)
        # No angle change (d_delta = 0)
        # No acceleration (d_omega = 0)
        # Balanced Forces (P_mech = P_elec)
        
        omega = torch.tensor([0.0])
        d_delta_dt = torch.tensor([0.0])
        d_omega_dt = torch.tensor([0.0])
        p_mech = torch.tensor([1.0])
        p_elec = torch.tensor([1.0])

        # 3. Compute Residual (The error in the law)
        res_delta, res_omega = swing(omega, d_delta_dt, d_omega_dt, p_mech, p_elec)

        # 4. Assert Compliance
        # Ideally 0.0, allowing small float tolerance
        assert torch.isclose(res_delta, torch.tensor(0.0), atol=1e-6), "Kinematic violation detected"
        assert torch.isclose(res_omega, torch.tensor(0.0), atol=1e-6), "Newton's Law violation detected"

    def test_swing_equation_dynamics(self):
        """
        Law: Dynamics (F = ma).
        Scenario: Loss of Load (P_elec drops by 50%).
        Expectation: Generator MUST accelerate. If we claim it doesn't, the residual should spike.
        """
        swing = SwingEquation(inertia_h=2.0, damping_d=0.0) # Simplify: No friction

        # Unbalance: P_mech (1.0) > P_elec (0.5) -> Net Force = 0.5
        # Law: 2H * d_omega_dt = Net_Force
        # 4.0 * d_omega_dt = 0.5 => d_omega_dt should be 0.125
        
        # We simulate a model predicting WRONG acceleration (0.0)
        res_delta, res_omega = swing(
            omega=torch.tensor([0.0]),
            d_delta_dt=torch.tensor([0.0]),
            d_omega_dt=torch.tensor([0.0]), # PHYSICS VIOLATION: Claiming no acceleration despite force
            p_mech=torch.tensor([1.0]),
            p_elec=torch.tensor([0.5])
        )

        # Residual = Inertial_Force - Net_Force
        # Residual = 0 - 0.5 = -0.5
        assert torch.isclose(res_omega, torch.tensor(-0.5), atol=1e-5)
        print("\n✅ Dynamics Test: Physics engine correctly flagged 'Energy Creation' violation.")

    def test_power_flow_kirchhoff(self):
        """
        Law: Kirchhoff's Current Law (AC Power Flow).
        Scenario: Simple 2-Bus System (Source -> Load).
        Expectation: Power calculated by physics formula matches injected power.
        """
        num_buses = 2
        pf = PowerFlowEquation(num_buses)

        # 1. Define Topology (Y-Bus Matrix)
        # Line with Impedance Z = j0.1 (Pure Inductance)
        # Admittance Y = 1/Z = -j10
        # B matrix captures this:
        # B = [[-10, 10], [10, -10]]
        G = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        B = torch.tensor([[-10.0, 10.0], [10.0, -10.0]]) 
        
        # Load topology into equation module
        Y_bus = torch.complex(G, B)
        pf.set_admittance_matrix(Y_bus)

        # 2. Define State
        # Bus 1: 1.0 p.u. voltage, 0.0 angle
        # Bus 2: 1.0 p.u. voltage, -0.1 rad angle (Lagging = Receiving Power)
        V_mag = torch.tensor([[1.0, 1.0]]) # Batch of 1
        V_ang = torch.tensor([[0.0, -0.1]])

        # 3. Analytic Solution (Textbook Formula)
        # P = (V1*V2/X) * sin(delta)
        # P = (1*1/0.1) * sin(0.1) = 10 * 0.0998 = 0.998
        expected_p_flow = 1.0 * 1.0 * 10.0 * np.sin(0.1)
        
        # Bus 1 sends, Bus 2 receives
        P_in = torch.tensor([[expected_p_flow, -expected_p_flow]])
        Q_in = torch.zeros_like(P_in) # Ignore reactive for simplicity

        # 4. Compute Residual
        res_P, res_Q = pf(V_mag, V_ang, P_in, Q_in)

        # 5. Assertions
        # The physics module should agree with the analytical formula
        assert torch.max(torch.abs(res_P)) < 1e-5, f"Power Flow mismatch: {res_P}"

    def test_battery_thermal_laws(self):
        """
        Law: First Law of Thermodynamics (Heat Generation).
        Scenario: Rapid charging (High Current).
        Expectation: Battery temperature MUST rise. Constant temp is physically impossible.
        """
        # Setup: 1kg cell, R=0.1 Ohm, Adiabatic (No cooling)
        thermal = BatteryThermalEquation(
            mass_kg=1.0, 
            specific_heat_cp=1000.0, 
            internal_resistance_r=0.1, 
            heat_transfer_coeff_h=0.0, 
            surface_area_a=1.0
        )

        current = torch.tensor([100.0]) # 100 Amps
        
        # Heat Gen Q = I^2 * R = 10000 * 0.1 = 1000 Watts
        # Rise Rate = Q / (m*Cp) = 1000 / 1000 = 1.0 degC/sec
        
        # Case A: Correct Physics (Temp rising at 1.0 C/s)
        res_valid = thermal(
            temp_c=torch.tensor([25.0]), 
            d_temp_dt=torch.tensor([1.0]), 
            current_a=current, 
            temp_amb_c=torch.tensor([25.0])
        )
        assert torch.isclose(res_valid, torch.tensor(0.0), atol=1e-5)

        # Case B: AI Hallucination (Claiming temp is stable during charging)
        res_hallucination = thermal(
            temp_c=torch.tensor([25.0]), 
            d_temp_dt=torch.tensor([0.0]), # Violation!
            current_a=current, 
            temp_amb_c=torch.tensor([25.0])
        )
        
        # Residual should capture the missing 1000W of heat
        assert torch.abs(res_hallucination) > 100.0
      
