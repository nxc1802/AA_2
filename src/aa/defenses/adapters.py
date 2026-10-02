import torch
import torch.nn as nn
from typing import Optional

from aa.utils import prepare_model_for_eval
from aa.defenses.bpda import BPDAFunction


class DefendedModelAdapter(nn.Module):
    """Wraps a base model with preprocessing defense in adaptive (BPDA) or oblivious mode."""
    def __init__(self, model: nn.Module, defense=None, mode: str = "adaptive"):
        super().__init__()
        self.model = prepare_model_for_eval(model)
        self.defense = defense
        self.mode = mode.lower()
        assert self.mode in ["adaptive", "oblivious"], f"Invalid mode: {mode}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.defense is None:
            return self.model(x)

        if self.mode == "adaptive":
            if getattr(self.defense, "is_differentiable", False):
                defended_x = self.defense.defend(x)
            else:
                defended_x = BPDAFunction.apply(x, self.defense)
            return self.model(defended_x.contiguous())
        else:
            return self.model(x)

    def evaluate_defended(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates model prediction on defended input (used for evaluating oblivious attacks)."""
        if self.defense is None:
            return self.model(x)
        defended_x = self.defense.defend(x)
        return self.model(defended_x.contiguous())
