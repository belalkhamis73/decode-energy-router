from .base import ModelPlugin
import torch

class SolarModelPlugin(ModelPlugin):
    @property
    def name(self) -> str:
        return "solar"
    
    def build_model(self, input_dim=4, hidden_dim=64, output_dim=1, **kwargs):
        # NOTE: Ensure deeponet is available in ml_models.architectures
        from ml_models.architectures.deeponet import DeepONet
        return DeepONet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    def prepare_input(self, ghi, temp, panel_angle, cloud_override=None, **kwargs):
        cloud_factor = cloud_override if cloud_override is not None else 0.0
        return torch.tensor([[ghi, temp, panel_angle, cloud_factor]], dtype=torch.float32)
