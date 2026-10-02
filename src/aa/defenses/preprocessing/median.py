import torch
import torch.nn.functional as F
from aa.defenses.base import BaseDefense


def median_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Applies Median Filter preprocessing to tensor x (B, C, H, W).
    Uses reflect padding to avoid artificial dark border artifacts on image edges.
    """
    padding = kernel_size // 2
    x_padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    B, C, H, W = x.shape
    unfolded = F.unfold(x_padded, kernel_size=kernel_size, padding=0)
    unfolded = unfolded.reshape(B, C, kernel_size * kernel_size, H * W)
    filtered = unfolded.median(dim=2).values
    return filtered.reshape(B, C, H, W)


class MedianFilterDefense(BaseDefense):
    is_differentiable = False

    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return median_filter(x, kernel_size=self.kernel_size)
