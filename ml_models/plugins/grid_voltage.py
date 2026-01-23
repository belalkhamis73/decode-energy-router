from .base import ModelPlugin
import torch

class GridVoltageModelPlugin(ModelPlugin):
    @property
    def name(self) -> str:
        return "grid_voltage"
    
    def build_model(self, input_dim, output_dim, n_buses, hidden_dim=64, **kwargs):
        from ml_models.architectures.deeponet import DeepONet
        # Passing n_buses explicitly as grid models often differ in architecture
        return DeepONet(n_buses=n_buses, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    def prepare_input(self, load, generation, topology=None, **kwargs):
        # Concatenate topology vector if provided
        if topology is not None:
            input_tensor = torch.cat([load, generation, topology], dim=-1)
        else:
            input_tensor = torch.cat([load, generation], dim=-1)
        
        # Ensure batch dimension exists
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        return input_tensor
