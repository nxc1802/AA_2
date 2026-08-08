import torch
import torch.nn as nn
from src.core.utils import prepare_model_for_eval

class BPDAFunction(torch.autograd.Function):
    """
    Backward Pass Differentiable Approximation (BPDA) for non-differentiable preprocessing defenses.
    Forward pass: applies defense transformation D(x).
    Backward pass: identity gradient approximation dD(x)/dx = I (Straight-Through Estimator).
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, defense_obj) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return defense_obj.defend(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-Through Estimator: pass gradient through unchanged
        return grad_output, None


class DefendedModelAdapter(nn.Module):
    """
    Adapter wrapping a base classifier model and a preprocessing defense.
    Supports both Adaptive Evaluation (gradient flows through BPDA/differentiable defense)
    and Oblivious Evaluation (attack generated against undefended model).
    """
    def __init__(self, model: nn.Module, defense=None, mode: str = "adaptive"):
        super().__init__()
        self.model = prepare_model_for_eval(model)
        self.defense = defense
        self.mode = mode.lower()
        assert self.mode in ["adaptive", "oblivious"], f"Unsupported defense evaluation mode: {mode}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.defense is None:
            return self.model(x)

        if self.mode == "adaptive":
            # Differentiable defense (e.g. Gaussian Blur) vs BPDA (e.g. Median Filter, JPEG, TVM)
            if getattr(self.defense, "is_differentiable", False):
                defended_x = self.defense.defend(x)
            else:
                defended_x = BPDAFunction.apply(x, self.defense)
            return self.model(defended_x)
        else:
            # Oblivious mode: undefended forward pass for attacker
            return self.model(x)
