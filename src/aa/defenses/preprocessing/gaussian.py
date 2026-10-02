import torch
import torch.nn.functional as F
from aa.defenses.base import BaseDefense


def gaussian_blur(x: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Applies Gaussian Blur preprocessing to tensor x (B, C, H, W).
    Uses reflect padding to avoid artificial dark border artifacts on image edges.
    """
    channels = x.size(1)
    radius = kernel_size // 2
    kernel_1d = torch.exp(-torch.arange(-radius, radius + 1, device=x.device).float()**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_4d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    x_padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(x_padded, kernel_4d, padding=0, groups=channels)


class GaussianBlurDefense(BaseDefense):
    is_differentiable = True

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
