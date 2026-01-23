from .base import ModelPlugin
import torch

class LoadForecastModelPlugin(ModelPlugin):
    @property
    def name(self) -> str:
        return "load_forecast"
    
    def build_model(self, input_dim, output_dim, hidden_dim=64, **kwargs):
        from ml_models.architectures.deeponet import DeepONet
        return DeepONet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    def prepare_input(self, historical_window, weather, time_features, **kwargs):
        input_tensor = torch.cat([
            historical_window.flatten(),
            weather.flatten(),
            time_features.flatten()
        ]).unsqueeze(0)
        return input_tensor
