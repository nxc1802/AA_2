import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from PIL import Image
import numpy as np

from aa.utils import prepare_model_for_eval


def gaussian_blur(x: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Applies Gaussian Blur preprocessing to tensor x (B, C, H, W)."""
    channels = x.size(1)
    radius = kernel_size // 2
    kernel_1d = torch.exp(-torch.arange(-radius, radius + 1, device=x.device).float()**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_4d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel_4d, padding=radius, groups=channels)


def median_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Applies Median Filter preprocessing to tensor x (B, C, H, W)."""
    padding = kernel_size // 2
    unfolded = F.unfold(x, kernel_size=kernel_size, padding=padding)
    B, C, H, W = x.shape
    unfolded = unfolded.view(B, C, kernel_size * kernel_size, H * W)
    filtered = unfolded.median(dim=2).values
    return filtered.view(B, C, H, W)


def jpeg_compression(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Applies JPEG compression to tensor x (B, C, H, W) in range [0, 1]."""
    device = x.device
    np_imgs = (x.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
    defended_np = []
    for img_np in np_imgs:
        pil_img = Image.fromarray(img_np)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        reconstructed = Image.open(buf).convert("RGB")
        defended_np.append(np.array(reconstructed).astype(np.float32) / 255.0)
    defended_tensor = torch.from_numpy(np.stack(defended_np)).permute(0, 3, 1, 2).to(device)
    return defended_tensor


def total_variation_minimization(x: torch.Tensor, iters: int = 5, step_size: float = 0.05) -> torch.Tensor:
    """Applies Total Variation Minimization smoothing to tensor x."""
    x_def = x.clone().detach()
    for _ in range(iters):
        diff_h = torch.abs(x_def[:, :, 1:, :] - x_def[:, :, :-1, :])
        diff_w = torch.abs(x_def[:, :, :, 1:] - x_def[:, :, :, :-1])
        grad_h = F.pad(diff_h, (0, 0, 1, 0)) - F.pad(diff_h, (0, 0, 0, 1))
        grad_w = F.pad(diff_w, (1, 0, 0, 0)) - F.pad(diff_w, (0, 1, 0, 0))
        x_def = torch.clamp(x_def - step_size * (grad_h + grad_w), 0.0, 1.0)
    return x_def


class GaussianBlurDefense:
    is_differentiable = True

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)


class MedianFilterDefense:
    is_differentiable = False

    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return median_filter(x, kernel_size=self.kernel_size)


class JPEGDefense:
    is_differentiable = False

    def __init__(self, quality: int = 75):
        self.quality = quality

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return jpeg_compression(x, quality=self.quality)


class TVMDefense:
    is_differentiable = False

    def __init__(self, iters: int = 5, step_size: float = 0.05):
        self.iters = iters
        self.step_size = step_size

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return total_variation_minimization(x, iters=self.iters, step_size=self.step_size)


class BPDAFunction(torch.autograd.Function):
    """Straight-Through Estimator (STE) / BPDA pass for non-differentiable defenses."""
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, defense_obj) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return defense_obj.defend(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


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
            return self.model(defended_x)
        else:
            return self.model(x)
