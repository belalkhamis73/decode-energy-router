"""
D.E.C.O.D.E. CLI Runner (v3.0 - Active Router Edition)
Verification Tool for Multi-Source Dispatch & Financial KPIs.
"""

import requests
import time
import sys
import random
from typing import Dict, Any

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}

def print_header():
    print("\n" + "="*95)
    print(" 🚀  D.E.C.O.D.E.  |  Active Energy Router  |  IEEE 118 Benchmark")
    print("="*95 + "\n")

def run_cli():
    print_header()

    # --- 1. Connection Check ---
    print("[System] Connecting to Backend API...")
    try:
        health = requests.get(f"{API_BASE_URL}/", timeout=2)
        if health.status_code == 200:
            print(f"   ✅ Backend Online: {health.json().get('status', 'OK')}")
        else:
            print("   ❌ Backend Error: System returned unhealthy status.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Failed! Ensure backend is running at {API_BASE_URL}")
        sys.exit(1)

    # --- 2. Operational Context ---
    print("\n[Step 1] Operational Context")
    print("   Initializing 'Egypt Summer' Scenario on IEEE 118-Bus Grid...")
    
    config_payload = {
        "grid_topology": "ieee118",
        "weather_profile": "solar_egypt"
    }
    
    try:
        resp = requests.post(f"{API_BASE_URL}/context/configure", json=config_payload, headers=HEADERS)
        resp.raise_for_status()
        session_data = resp.json()
        session_id = session_data["session_id"]
        
        print(f"   ✅ Session Created: ID {session_id}")
        grid_meta = session_data.get('grid_summary', {})
        # Verify correct grid loading (118 vs 14)
        bus_count = grid_meta.get('buses', 'N/A')
        print(f"   ℹ️  Topology: {bus_count} Buses (High Voltage Transmission)")
        
    except Exception as e:
        print(f"❌ Configuration Failed: {e}")
        sys.exit(1)

    # --- 3. Training Phase ---
    print(f"\n[Step 2] Physics-Informed Training")
    print("   Training DeepONet on localized data...")
    
    train_payload = {"session_id": session_id, "epochs": 5}
    requests.post(f"{API_BASE_URL}/model/train", json=train_payload, headers=HEADERS)
    
    # Poll for completion with Graceful Error Handling
    while True:
        status_resp = requests.get(f"{API_BASE_URL}/model/status/{session_id}")
        state_data = status_resp.json()
        status = state_data["status"]
        
        if status == "READY":
            # ROBUST ACCESS: Use .get() to avoid KeyError if metrics missing
            metrics = state_data.get("metrics", {})
            loss = metrics.get("final_loss", 0.0)
            print(f"   ✅ Model Converged. Physics Residual: {loss:.5f}")
            break
        elif status == "FAILED":
            print("   ❌ Training Failed.")
            sys.exit(1)
        time.sleep(0.5)

    # --- 4. Live Simulation (The 'All System' View) ---
    print(f"\n[Step 3] Real-Time Active Dispatch (24-Hour Horizon)")
    print("   Monitoring Multi-Source Energy Router (Solar, Wind, Battery, V2G, Diesel)...")
    time.sleep(1)
    
    # Updated Header for New Fields
    header = "{:<4} | {:<8} | {:<14} | {:<8} | {:<8} | {:<8} | {:<8}"
    print("-" * 90)
    print(header.format("HR", "VOLTAGE", "ACTION", "RENEW", "BATT", "DIESEL", "V2G"))
    print("-" * 90)

    for tick in range(0, 24, 2): # Every 2 hours
        # Simulating load variation (Peak at evening)
        load_scaling = 1.0 + random.uniform(-0.1, 0.2)
        if 18 <= tick <= 21: load_scaling = 1.4 

        predict_payload = {
            "session_id": session_id, 
            "tick": tick, 
            "load_scaling": load_scaling
        }
        
        try:
            resp = requests.post(f"{API_BASE_URL}/simulation/predict", json=predict_payload, headers=HEADERS)
            data = resp.json()
            
            # Extract Data safely
            grid = data.get("grid_state", {})
            dispatch = data.get("dispatch", {})
            kpis = data.get("kpis", {}) # Financial Engine Output
            
            volts = grid.get("avg_voltage_pu", 0.0)
            action = dispatch.get("action", "IDLE")
            
            # Asset Values (kW)
            solar_kw = dispatch.get("solar_kw", 0.0) + dispatch.get("wind_kw", 0.0)
            batt_kw = dispatch.get("battery_kw", 0.0) # Neg = Discharge
            diesel_kw = dispatch.get("diesel_kw", 0.0)
            v2g_kw = dispatch.get("v2g_kw", 0.0)
            
            # Formatting
            v_str = f"{volts:.3f} pu"
            
            # Color Logic for Dashboard
            row_color = "\033[0m" # Default White
            if diesel_kw > 0:
                row_color = "\033[91m" # Red (Burning Fuel)
            elif batt_kw < 0 or v2g_kw < 0:
                row_color = "\033[93m" # Yellow (Discharging Storage)
            elif solar_kw > 100:
                row_color = "\033[92m" # Green (High Renewables)
                
            print(f"{row_color}{tick:<4} | {v_str:<8} | {action:<14} | {solar_kw:<8.0f} | {batt_kw:<8.0f} | {diesel_kw:<8.0f} | {v2g_kw:<8.0f}\033[0m")
            
        except Exception as e:
            print(f"❌ Error at tick {tick}: {e}")
        
        time.sleep(0.3)

    print("-" * 90)
    print("🏁 Simulation Complete.")

if __name__ == "__main__":
    run_cli()    
    choice = input(f"   > Select option (default '{default_key}'): ").strip().lower()
    
    if choice == "":
        return default_key
    if choice in options:
        return choice
    
    print(f"   ⚠️  Invalid choice. Using default: {default_key}")
    return default_key

def spinner(text: str, duration: int = 2):
    """Visual feedback for waiting actions."""
    chars = "|/-\\"
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        sys.stdout.write(f"\r   {text} {chars[i % len(chars)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " "*len(text)*2 + "\r")

# --- Main Execution Flow ---

def run_cli():
    print_header()

    # --- 1. Connection Check ---
    print("[System] Connecting to Backend API...")
    try:
        health = requests.get(f"{API_BASE_URL}/", timeout=2)
        if health.status_code == 200:
            print(f"   ✅ Backend Online: {health.json()['system']}")
        else:
            print("   ❌ Backend Error: System returned unhealthy status.")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Failed! Ensure backend is running at {API_BASE_URL}")
        print("   💡 Tip: Run 'uvicorn backend.main:app --reload' in a separate terminal.")
        sys.exit(1)

    # --- 2. User Inputs (Context Selection) ---
    print("\n[Step 1] Configure Digital Twin Context")
    print("Select the operational environment for the simulation.")
    
    # Grid Selection
    grid_choice = get_user_selection(
        "Choose Grid Topology:",
        {
            "14": "IEEE 14-Bus (Standard Microgrid)", 
            "118": "IEEE 118-Bus (Large Scale Benchmark)"
        },
        "118"
    )
    # Map user choice to API parameter
    topology_map = {"14": "ieee14", "118": "ieee118"}

    # Weather Selection
    weather_choice = get_user_selection(
        "Choose Weather Profile:",
        {
            "solar": "Egypt/Suez (High Solar Irradiance)", 
            "wind": "North Sea (High Wind Volatility)"
        },
        "solar"
    )
    # Map user choice to API parameter
    weather_map = {"solar": "solar_egypt", "wind": "wind_north"}

    # --- 3. Session Initialization ---
    config_payload = {
        "grid_topology": topology_map[grid_choice],
        "weather_profile": weather_map[weather_choice]
    }
    
    print(f"\n🔄 Initializing Session for {config_payload['grid_topology'].upper()}...")
    
    try:
        resp = requests.post(f"{API_BASE_URL}/context/configure", json=config_payload, headers=HEADERS)
        resp.raise_for_status()
        session_data = resp.json()
        session_id = session_data["session_id"]
        
        print(f"   ✅ Session Created: ID {session_id}")
        grid_meta = session_data.get('grid_summary', {})
        print(f"   ℹ️  Grid Specs: {grid_meta.get('buses', 'N/A')} Buses | {grid_meta.get('lines', 'N/A')} Lines")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Configuration Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"   Server says: {e.response.text}")
        sys.exit(1)

    # --- 4. Training Phase ---
    print(f"\n[Step 2] Training Physics-Informed Core")
    print("The system will now train the DeepONet operator specific to your selected grid.")
    
    epochs_input = input("   > Enter training epochs (default 50): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 50
    
    train_payload = {"session_id": session_id, "epochs": epochs}
    
    try:
        requests.post(f"{API_BASE_URL}/model/train", json=train_payload, headers=HEADERS)
        print("   🚀 Training Job Dispatched...")
        
        # Polling Loop
        start_time = time.time()
        while True:
            status_resp = requests.get(f"{API_BASE_URL}/model/status/{session_id}")
            state_data = status_resp.json()
            status = state_data["status"]
            
            if status == "READY":
                loss = state_data["metrics"].get("final_loss", 0.0)
                duration = time.time() - start_time
                print(f"\n   ✅ Model Converged in {duration:.1f}s. Final Physics Loss: {loss:.5f}")
                break
            elif status == "FAILED":
                print("\n   ❌ Training Failed on Server.")
                print(f"   Reason: {state_data.get('error', 'Unknown')}")
                sys.exit(1)
            else:
                sys.stdout.write(f"\r   ⏳ Status: {status}...")
                sys.stdout.flush()
                time.sleep(1)
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Training Communication Error: {e}")
        sys.exit(1)

    # --- 5. Live Simulation Phase ---
    print(f"\n[Step 3] Live Operation (24-Hour Cycle)")
    print("   Simulating real-time data streaming & active control...")
    print("   Press Ctrl+C to stop early.")
    time.sleep(1)
    
    # Table Header
    # Format: Tick | Input | Voltage | Action | Status
    header_fmt = "{:<5} | {:<15} | {:<15} | {:<20} | {:<15}"
    print("-" * 85)
    print(header_fmt.format("TICK", "INPUT (W/m²)", "GRID VOLTAGE", "CONTROL ACTION", "SAFETY STATUS"))
    print("-" * 85)

    try:
        for tick in range(24):
            # Simulate dynamic load scaling (e.g., peak demand in evening)
            # This simulates "User behavior" changing throughout the day
            load_scaling = 1.0
            if 17 <= tick <= 21: # Evening peak
                load_scaling = 1.3
            elif 1 <= tick <= 5: # Night trough
                load_scaling = 0.7
            
            # Add some randomness to load
            load_scaling += random.uniform(-0.05, 0.05)

            predict_payload = {
                "session_id": session_id, 
                "tick": tick, 
                "load_scaling": load_scaling
            }
            
            # API Call
            resp = requests.post(f"{API_BASE_URL}/simulation/predict", json=predict_payload, headers=HEADERS)
            data = resp.json()
            
            # Parsing Response for Display
            inputs = data.get("inputs", {})
            solar_input = inputs.get("solar_w_m2", 0.0)
            if "wind" in weather_choice:
                # If wind was chosen, display wind speed instead
                solar_input = inputs.get("wind_m_s", 0.0)
                input_label = f"{solar_input:.1f} m/s"
            else:
                input_label = f"{solar_input:.1f} W/m²"
            
            grid_state = data.get("grid_state", {})
            voltage = grid_state.get("avg_voltage_pu", 1.0)
            action = data.get("control_decision", "IDLE")
            
            # Safety Status Logic
            safety_info = data.get("safety_system", {})
            if safety_info.get("physics_violation_detected", False):
                # Highlight violations in Yellow/Warning text
                safety_msg = "⚠️ VIOLATION (Fixed)" 
                color_start = "\033[93m" # Yellow
                color_end = "\033[0m"
                row_str = header_fmt.format(tick, input_label, f"{voltage:.4f} p.u.", action, safety_msg)
                print(f"{color_start}{row_str}{color_end}")
            else:
                safety_msg = "✅ STABLE"
                print(header_fmt.format(tick, input_label, f"{voltage:.4f} p.u.", action, safety_msg))
            
            # Pace the output for readability
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 User aborted simulation.")
    except Exception as e:
        print(f"\n❌ Simulation Error: {e}")

    print("-" * 85)
    print("🏁 24-Hour Simulation Complete.")

if __name__ == "__main__":
    run_cli()
  

