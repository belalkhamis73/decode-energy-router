"""
Integration Tests for the Virtual Grid (Digital Twin).
Verifies the end-to-end flow of data from API -> AI Model -> Physics Validation.

Principles:
- Black Box Testing: Tests the system from the outside (API surface).
- Realism: Uses valid Pydantic payloads matching the 'Track A' competition specs.
- Resilience: Specifically tests "Black Swan" failure modes.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

# Import the application assembly
from backend.app.main import app
# We explicitly import the router to ensure it's tested even if main.py isn't fully wired yet
from backend.app.api.routes import simulation

# Mount the simulation router for testing purposes if not already in main.py
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["simulation"])

# Initialize the Test Client (Synchronous wrapper for Async FastAPI)
client = TestClient(app)

class TestDigitalTwinAPI:
    
    def test_system_heartbeat(self):
        """
        Scenario: SCADA system pings the Digital Twin.
        Expectation: 200 OK and status 'healthy'.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["mode"] == "physics-informed"

    def test_solar_state_prediction_physics_compliance(self):
        """
        Scenario: Control loop requests next-step GHI forecast.
        Physics Check: The AI must not return negative solar irradiance.
        """
        # 1. Construct valid 24-hour telemetry window (normalized)
        # [DNI, DHI, Temp, Wind, CosZenith]
        fake_window = [[0.8, 0.2, 25.0, 5.0, 0.9] for _ in range(24)]
        
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "telemetry_window": fake_window
        }

        # 2. Fire Request
        response = client.post("/predict/solar-state", json=payload)
        
        # 3. Assertions
        assert response.status_code == 200, f"Inference failed: {response.text}"
        data = response.json()
        
        # Check Data Contract
        assert "ghi_forecast" in data
        assert "confidence_score" in data
        
        # Check Physics (Energy Conservation)
        # The model should generally output normalized values [0, 1] or positive floats
        assert data["ghi_forecast"] >= 0.0, "Physics Violation: Negative Solar Generation detected!"

    def test_simulation_fault_injection(self):
        """
        Scenario: Operator runs a 'What-If' contingency analysis (Line Trip).
        Expectation: The physics engine runs and returns a stable or unstable grid state.
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_component": "Line_Bus2_Bus4",
            "component_type": "line",
            "fault_type": "3ph_short",
            "magnitude": 0.0, # 0.0 means complete short circuit (0 voltage)
            "duration_ms": 150
        }

        response = client.post("/api/v1/simulation/run", json=payload)
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify the structure matches our Physics Schema
        assert "grid_state" in result
        assert len(result["grid_state"]["voltages"]) > 0
        assert isinstance(result["physics_violation_detected"], bool)

    def test_fail_fast_on_causality_violation(self):
        """
        Scenario: Malformed request with negative duration (Time travel).
        Expectation: 422 Unprocessable Entity (Validation Error).
        Principle: Fail Fast - Don't let bad data touch the physics engine.
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_component": "Gen_1",
            "component_type": "gen",
            "fault_type": "gen_trip",
            "magnitude": 1.0,
            "duration_ms": -500 # Violation: Negative Time
        }

        response = client.post("/api/v1/simulation/run", json=payload)
        
        # Should be blocked by Pydantic before reaching logic
        assert response.status_code == 422 
        assert "Input should be greater than or equal to 1" in response.text

    def test_stress_test_queueing(self):
        """
        Scenario: Triggering a massive async batch job.
        Expectation: 202 Accepted (Non-blocking).
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_component": "All_Assets",
            "component_type": "weather",
            "fault_type": "cloud_gorilla",
            "magnitude": 0.8,
            "duration_ms": 5000
        }
        
        response = client.post("/api/v1/simulation/stress-test/async", json=payload)
        
        assert response.status_code == 202
        assert "task_id" in response.json()
      
