from abc import ABC, abstractmethod
import torch

class BaseDefense(ABC):
    """Abstract base class for all defense transformations."""
    is_differentiable: bool = False

    @abstractmethod
    def defend(self, x: torch.Tensor) -> torch.Tensor:
        """Applies defense transformation to tensor x in range [0, 1]."""
        pass
