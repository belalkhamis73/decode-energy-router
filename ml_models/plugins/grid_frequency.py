from .base import ModelPlugin
import torch

class GridFrequencyModelPlugin(ModelPlugin):
    @property
    def name(self) -> str:
        return "grid_frequency"
    
    def build_model(self, input_dim=2, hidden_dim=64, output_dim=1, **kwargs):
        from ml_models.architectures.deeponet import DeepONet
        return DeepONet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    def prepare_input(self, power_imbalance, inertia, **kwargs):
        return torch.tensor([[power_imbalance, inertia]], dtype=torch.float32)
