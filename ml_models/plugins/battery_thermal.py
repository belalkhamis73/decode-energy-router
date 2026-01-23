from .base import ModelPlugin
import torch

class BatteryThermalModelPlugin(ModelPlugin):
    @property
    def name(self) -> str:
        return "battery_thermal"
    
    def build_model(self, input_dim=4, hidden_dim=64, output_dim=1, **kwargs):
        from ml_models.architectures.deeponet import DeepONet
        return DeepONet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    def prepare_input(self, soc, current, temp_ambient, cooling_factor=1.0, **kwargs):
        return torch.tensor([[soc, current, temp_ambient, cooling_factor]], dtype=torch.float32)
