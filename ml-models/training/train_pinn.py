"""
PINN Training Engine.
Refactored for SaaS: Encapsulates training logic into a session-based function.
Responsible for initializing, training, and returning the DeepONet model 
based on dynamic user inputs (Topology + Weather) passed from the Backend.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Import the architecture (Assumes the file exists as per previous steps)
from ml_models.architectures.deeponet import DeepONet

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PINN_Trainer")

def train_session_model(
    dataset: Dict[str, Any], 
    hyperparams: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Main entry point for the Backend to train a Digital Twin.
    
    Args:
        dataset: Dictionary containing 'topology' (Grid specs) and 'weather' (Context).
                 Source: backend.services.data_manager
        hyperparams: Dictionary containing 'epochs', 'learning_rate', 'device'.
        
    Returns:
        model: The trained PyTorch DeepONet model (in-memory).
        metrics: Dictionary of final performance metrics (e.g., loss).
    """
    
    # 1. Setup Configuration
    epochs = hyperparams.get("epochs", 50)
    lr = hyperparams.get("learning_rate", 1e-3)
    device = hyperparams.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"⚙️  Initializing Training Session on {device.upper()}")
    
    # 2. Data Preparation (Tensorization)
    # We unpack the dictionary passed from DataManager
    topology = dataset["topology"]
    weather = dataset["weather"]
    
    # Input Function 'u' (Context): [Solar, Wind, Temp]
    # Weather data is a list of floats -> Convert to Tensor [24, 3]
    weather_matrix = np.stack([
        weather["ghi"], 
        weather["wind_speed"], 
        weather["temperature"]
    ], axis=1)
    
    u_train = torch.tensor(weather_matrix, dtype=torch.float32).to(device)
    
    # Domain Variables 'y' (Locations): Bus IDs
    # DeepONet needs to know WHICH buses to predict for.
    # We train on ALL buses simultaneously.
    n_buses = topology["n_buses"]
    
    # Create a batch of Bus IDs [24, N_Buses]
    # We repeat the bus indices for every time step
    bus_ids = torch.arange(n_buses, dtype=torch.long).unsqueeze(0).repeat(24, 1).to(device)
    
    # Target 'G(u)(y)': Ideal Voltage Profile (Stability Goal)
    # In a real scenario, this would come from a Power Flow solver.
    # For the SaaS Demo/MVP, we train the model to maintain 1.0 p.u. (Ideal Stability).
    target_voltage = torch.ones(24, n_buses, dtype=torch.float32).to(device)
    
    # 3. Model Initialization
    # alteration: Passing 'n_buses' dynamically to resize the Trunk Net
    model = DeepONet(
        input_dim=3,        # [GHI, Wind, Temp]
        hidden_dim=64,
        output_dim=1,       # Voltage Magnitude
        n_buses=n_buses     # <--- Critical Alteration for IEEE 118 support
    ).to(device)
    
    model.train()
    
    # 4. Optimization Setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # Physics Loss can be added here as a regularization term
    
    # 5. Training Loop
    logger.info(f"🚀 Starting Loop: {epochs} Epochs | Grid: {topology['name']}")
    final_loss = 0.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Pass: DeepONet predicts Voltage(t, bus) based on Weather(t)
        # u_train: [24, 3]
        # bus_ids: [24, n_buses]
        prediction = model(u_train, bus_ids) # Output: [24, n_buses]
        
        # Loss Calculation
        loss = criterion(prediction, target_voltage)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        final_loss = loss.item()
        
        # Logging (every 10% of progress)
        if epochs > 10 and epoch % (epochs // 10) == 0:
            logger.debug(f"   [Epoch {epoch}] Loss: {final_loss:.6f}")

    logger.info(f"✅ Training Complete. Final Loss: {final_loss:.5f}")
    
    # 6. Return Artifacts (Do not save to disk here; let Backend decide)
    metrics = {
        "final_loss": final_loss, 
        "iterations": epochs,
        "grid_size": n_buses
    }
    
    # Move model to CPU before returning to save GPU memory in session store
    return model.cpu(), metrics

# --- Unit Test (For debugging standalone) ---
if __name__ == "__main__":
    # Mock Data to verify logic without backend
    mock_topology = {"name": "test_grid", "n_buses": 5}
    mock_weather = {
        "ghi": [100.0] * 24,
        "wind_speed": [5.0] * 24,
        "temperature": [25.0] * 24
    }
    
    dataset = {"topology": mock_topology, "weather": mock_weather}
    params = {"epochs": 5, "learning_rate": 0.01}
    
    try:
        model, metrics = train_session_model(dataset, params)
        print(f"Test Successful: {metrics}")
    except Exception as e:
        print(f"Test Failed: {e}")
