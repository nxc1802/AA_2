import torch

class BPDAFunction(torch.autograd.Function):
    """Straight-Through Estimator (STE) / BPDA pass for non-differentiable defenses."""
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, defense_obj) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return defense_obj.defend(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None
