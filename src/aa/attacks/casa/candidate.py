import torch
import torch.nn.functional as F


def compute_box_aware_gain(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Computes linear potential gain in [0, 1]^3 box for each spatial pixel:
    A_i = sum_c |g_{ic}| * (1 - x_{ic} if g_{ic} > 0 else x_{ic})
    Shape: (B, 1, H, W)
    """
    pos_gain = F.relu(grad) * (1.0 - x)
    neg_gain = F.relu(-grad) * x
    gain_rgb = pos_gain + neg_gain
    return gain_rgb.sum(dim=1, keepdim=True)


class CandidateGenerator:
    """Manages candidate pool generation and dynamic refreshes."""
    def __init__(self, multiplier: int = 4):
        self.multiplier = multiplier

    def get_pool_size(self, effective_k: int, HW: int, fixed_size=None) -> int:
        if fixed_size is not None:
            M = max(fixed_size, effective_k + 16)
        else:
            M = max(64, effective_k * self.multiplier)
        return min(M, HW)
