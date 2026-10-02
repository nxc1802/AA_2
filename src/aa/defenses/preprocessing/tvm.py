import torch
import torch.nn.functional as F
from aa.defenses.base import BaseDefense


def total_variation_minimization(x: torch.Tensor, iters: int = 5, step_size: float = 0.05) -> torch.Tensor:
    """Applies Total Variation (TV) Minimization denoising to tensor x (B, C, H, W)."""
    x_def = x.clone().detach()
    for _ in range(iters):
        diff_h = x_def[:, :, 1:, :] - x_def[:, :, :-1, :]
        diff_w = x_def[:, :, :, 1:] - x_def[:, :, :, :-1]
        grad_h = F.pad(diff_h.sign(), (0, 0, 1, 0)) - F.pad(diff_h.sign(), (0, 0, 0, 1))
        grad_w = F.pad(diff_w.sign(), (1, 0, 0, 0)) - F.pad(diff_w.sign(), (0, 1, 0, 0))
        x_def = torch.clamp(x_def - step_size * (grad_h + grad_w), 0.0, 1.0)
    return x_def


class TVMDefense(BaseDefense):
    is_differentiable = False

    def __init__(self, iters: int = 5, step_size: float = 0.05):
        self.iters = iters
        self.step_size = step_size

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return total_variation_minimization(x, iters=self.iters, step_size=self.step_size)
