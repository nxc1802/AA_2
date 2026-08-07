import time
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class AttackResult:
    """Standardized result data container for adversarial attack evaluation."""
    adversarial: torch.Tensor
    success: torch.Tensor
    l0: torch.Tensor
    l2: torch.Tensor
    linf: torch.Tensor
    steps: torch.Tensor
    queries: torch.Tensor
    runtime_seconds: torch.Tensor
    best_loss: Optional[torch.Tensor] = None
