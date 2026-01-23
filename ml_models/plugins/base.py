from abc import ABC, abstractmethod
import torch

class ModelPlugin(ABC):
    """Abstract base class for model plugins"""
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model identifier (e.g., 'solar')"""
        pass

    @abstractmethod
    def build_model(self, **kwargs) -> torch.nn.Module:
        """Construct model architecture"""
        pass

    @abstractmethod
    def prepare_input(self, **kwargs) -> torch.Tensor:
        """Convert domain inputs to tensor"""
        pass
