from .base import ModelPlugin
import torch

class WindModelPlugin(ModelPlugin):
    @property
    def name(self) -> str:
        return "wind"
    
    def build_model(self, input_dim=3, hidden_dim=64, output_dim=1, **kwargs):
        from ml_models.architectures.deeponet import DeepONet
        return DeepONet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    def prepare_input(self, wind_speed, air_density=1.225, turbine_state=1, **kwargs):
        return torch.tensor([[wind_speed, air_density, float(turbine_state)]], dtype=torch.float32)
