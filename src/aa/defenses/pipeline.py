from typing import List
import torch
from aa.defenses.base import BaseDefense


class DefensePipeline(BaseDefense):
    """Composes multiple defenses sequentially."""
    def __init__(self, defenses: List[BaseDefense]):
        self.defenses = defenses
        self.is_differentiable = all(getattr(d, "is_differentiable", False) for d in defenses)

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for d in self.defenses:
            out = d.defend(out)
        return out
