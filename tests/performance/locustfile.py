"""
Performance Stress Test (Locust).
Simulates high-frequency SCADA telemetry and Operator control actions.
Target: Validate system stability under 'Storm Conditions' (1000+ concurrent requests).
"""

from locust import HttpUser, task, between, tag, events
import random
from datetime import datetime, timedelta
import json

# --- Physics Data Generators (prevent garbage-in/garbage-out) ---
def generate_telemetry_window():
    """Generates a realistic 24-hour sliding window (5 features)."""
    # Features: [DNI, DHI, Temp, Wind, CosZenith]
    # Physics constraint: DNI/DHI >= 0
    return [
        [
            random.uniform(0, 1000), # DNI
            random.uniform(0, 300),  # DHI
            random.uniform(10, 45),  # Temp
            random.uniform(0, 15),   # Wind
            random.uniform(-1, 1)    # CosZenith
        ]
        for _ in range(24)
    ]

class EnergyRouterUser(HttpUser):
    """
    Simulates a Grid Controller or IoT Gateway.
    Behavior: Constant polling of state + occasional heavy simulation requests.
    """
    # Real-world SCADA polls every 1-4 seconds
    wait_time = between(1, 4)

    @tag('core-path')
    @task(10) # High weight: This is the main heartbeat
    def push_telemetry_and_predict(self):
        """
        The 'Hot Path': Real-time inference request.
        Expected Latency: < 50ms
        """
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "telemetry_window": generate_telemetry_window()
        }
        
        with self.client.post("/predict/solar-state", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Prediction Failed: {response.text}")
            
            # SLA Check: Fail if inference is too slow for real-time control
            if response.elapsed.total_seconds() > 0.100:
                response.failure(f"SLA Violation: Latency {response.elapsed.total_seconds()}s > 100ms")

    @tag('simulation')
    @task(1) # Low weight: Occasional operator action
    def trigger_contingency_analysis(self):
        """
        The 'Heavy Path': CPU-bound Physics Simulation (Pandapower).
        """
        components = ["Line_1", "Bus_4", "Gen_2", "Transformer_Main"]
        faults = ["3ph_short", "1ph_ground", "gen_trip"]
        
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "target_component": random.choice(components),
            "component_type": "line", # Simplified for load test
            "fault_type": random.choice(faults),
            "magnitude": random.uniform(0.0, 1.0),
            "duration_ms": random.randint(50, 500)
        }

        # We expect this to be slower, so we don't enforce strict 100ms SLA
        self.client.post("/api/v1/simulation/run", json=payload)

    @tag('health')
    @task(2)
    def health_check(self):
        """K8s Liveness Probe simulation."""
        self.client.get("/health")

# --- Hooks for Custom Metrics ---
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("⚡ Starting Grid Stress Test...")
    print("   Target: Active Energy Router API")
    print("   Scenario: Storm Conditions (High Telemetry Load)")

@events.request.add_listener
def on_request(request_type, name,response_time, response_length, exception, **kwargs):
    """Log physics violations or crashes."""
    if exception:
        print(f"❌ Failure: {name} failed with {exception}")
  
