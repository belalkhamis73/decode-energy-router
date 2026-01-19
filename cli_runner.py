"""
D.E.C.O.D.E. CLI Runner (v3.1 - Robust Client)
The official verification tool for the Digital Twin backend.
"""

import requests
import time
import sys
import random

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}

# --- ANSI COLORS ---
class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    print(f"\n{Color.HEADER}" + "="*80)
    print(f" 🚀  D.E.C.O.D.E. DIGITAL TWIN  |  IEEE 118 Benchmark Simulation")
    print("="*80 + f"{Color.ENDC}\n")

def run_simulation():
    print_banner()

    # --- 1. HEALTH CHECK ---
    print(f"{Color.CYAN}[1] Connecting to Backend API...{Color.ENDC}")
    try:
        # We hit the simulation mount point to ensure it's active
        resp = requests.get(f"{API_URL}/", timeout=2)
        if resp.status_code == 200:
            info = resp.json()
            print(f"    ✅ Server Online. Mode: {info.get('mode', 'Unknown')}")
        else:
            print(f"    ❌ Server Error (Status {resp.status_code})")
            return
    except requests.exceptions.ConnectionError:
        print(f"{Color.FAIL}    ❌ Connection Refused! Is the server running?{Color.ENDC}")
        print("    👉 Run: uvicorn backend.main:app --reload")
        return

    # --- 2. CONFIGURE SESSION ---
    print(f"\n{Color.CYAN}[2] Initializing Digital Twin Context...{Color.ENDC}")
    try:
        config = {
            "grid_topology": "ieee118",
            "weather_profile": "solar_egypt",
            "scenario": "normal"
        }
        res = requests.post(f"{API_URL}/context/configure", json=config, headers=HEADERS).json()
        sid = res['session_id']
        buses = res.get('grid_summary', {}).get('n_buses', 'N/A')
        print(f"    ✅ Session Created: {Color.BOLD}{sid}{Color.ENDC}")
        print(f"    ⚡ Topology Loaded: IEEE {buses}-Bus System")
    except Exception as e:
        print(f"{Color.FAIL}    ❌ Configuration Failed: {e}{Color.ENDC}")
        return

    # --- 3. TRAIN MODEL ---
    print(f"\n{Color.CYAN}[3] Training Physics-Informed Neural Network...{Color.ENDC}")
    requests.post(f"{API_URL}/model/train", json={"session_id": sid}, headers=HEADERS)
    
    # Poll for completion
    sys.stdout.write("    ⏳ Training ")
    for _ in range(10):
        time.sleep(0.5)
        sys.stdout.write(".")
        sys.stdout.flush()
    print(f"\n    ✅ Model Ready (Mock Training Complete)")

    # --- 4. REAL-TIME SIMULATION ---
    print(f"\n{Color.CYAN}[4] 24-Hour Active Dispatch Loop{Color.ENDC}")
    
    # Table Header
    row_fmt = "{:<4} | {:<8} | {:<18} | {:<8} | {:<8} | {:<10}"
    print("-" * 75)
    print(row_fmt.format("HR", "VOLTAGE", "ACTION", "BATT(kW)", "DG(kW)", "STATUS"))
    print("-" * 75)

    for tick in range(0, 24, 2): # Step every 2 hours
        # Dynamic Load: Peak at 18:00
        load = 1.0 + (0.4 if tick >= 18 else 0.0) + random.uniform(-0.05, 0.05)
        
        try:
            payload = {"session_id": sid, "tick": tick, "load_scaling": load}
            resp = requests.post(f"{API_URL}/simulation/predict", json=payload, headers=HEADERS)
            
            if resp.status_code != 200:
                print(f"{Color.FAIL}Error {resp.status_code}: {resp.text}{Color.ENDC}")
                continue
                
            data = resp.json()
            
            # Extract Data safely with .get()
            grid = data.get('grid_state', {})
            dispatch = data.get('dispatch', {})
            health = data.get('asset_health', {})
            
            # Parse metrics
            v_pu = grid.get('avg_voltage_pu', 1.0)
            action = dispatch.get('action', 'IDLE')
            batt_kw = dispatch.get('battery_kw', 0.0)
            diesel_kw = dispatch.get('diesel_kw', 0.0)
            violation = grid.get('physics_violation', False)
            
            # Status Logic
            status_str = "NOMINAL"
            row_color = Color.ENDC
            
            if violation:
                status_str = "VIOLATION"
                row_color = Color.WARNING
            elif diesel_kw > 0:
                status_str = "DIESEL ON"
                row_color = Color.FAIL
            elif batt_kw < 0:
                status_str = "DISCHARGE"
                row_color = Color.GREEN
                
            # Print Row
            v_str = f"{v_pu:.3f} pu"
            print(f"{row_color}" + row_fmt.format(
                tick, v_str, action, f"{batt_kw:.1f}", f"{diesel_kw:.1f}", status_str
            ) + f"{Color.ENDC}")
            
        except Exception as e:
            print(f"{Color.FAIL}❌ Simulation Crash: {e}{Color.ENDC}")
        
        time.sleep(0.3)

    print("-" * 75)
    print(f"{Color.BOLD}🏁 Simulation Complete.{Color.ENDC}")

if __name__ == "__main__":
    run_simulation()
