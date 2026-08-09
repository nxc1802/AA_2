from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class AttackOutput:
    x_adv: torch.Tensor
    forward_evals: int = 0
    backward_evals: int = 0
    queries: int = 0
    metadata: Optional[Dict[str, Any]] = None


class Attack:
    """Base interface for all adversarial attacks."""
    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        raise NotImplementedError
