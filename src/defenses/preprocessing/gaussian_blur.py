import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianBlurDefense:
    """Applies Gaussian Blur filtering on input images as a defense."""
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def defend(self, images: torch.Tensor) -> torch.Tensor:
        device = images.device
        C = images.size(1)
        
        # Create Gaussian 1D kernel
        coords = torch.arange(self.kernel_size, dtype=torch.float32, device=device) - self.kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        
        # 2D Gaussian kernel
        kernel_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        
        padding = self.kernel_size // 2
        defended = F.conv2d(images, kernel, padding=padding, groups=C)
        return torch.clamp(defended, 0.0, 1.0)
