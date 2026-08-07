import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


def compute_per_sample_psnr(orig: torch.Tensor, adv: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Computes PSNR per sample in batch.
    
    Args:
        orig: Original images (B, C, H, W)
        adv: Adversarial images (B, C, H, W)
        max_val: Maximum pixel value range (default 1.0)
    Returns:
        Tensor (B,) of PSNR values in dB.
    """
    mse = torch.mean((orig - adv) ** 2, dim=[1, 2, 3])
    # Avoid log(0) when orig == adv
    mse_clamped = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse_clamped)
    # If identical, set PSNR to infinity/high float (e.g. 100.0)
    psnr = torch.where(mse < 1e-10, torch.tensor(100.0, device=orig.device), psnr)
    return psnr


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    gauss = torch.tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
        for x in range(window_size)
    ], device=device)
    gauss = gauss / gauss.sum()
    _1d_window = gauss.unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_per_sample_ssim(
    orig: torch.Tensor, 
    adv: torch.Tensor, 
    window_size: int = 11, 
    sigma: float = 1.5, 
    max_val: float = 1.0
) -> torch.Tensor:
    """
    Computes standard SSIM per sample in batch using Gaussian window.
    
    Args:
        orig: Original images (B, C, H, W)
        adv: Adversarial images (B, C, H, W)
        window_size: Gaussian kernel window size
        sigma: Standard deviation of Gaussian kernel
        max_val: Dynamic range of pixel values
    Returns:
        Tensor (B,) of SSIM indices in [0, 1].
    """
    B, C, H, W = orig.shape
    device = orig.device
    window = _gaussian_window(window_size, sigma, C, device)

    mu1 = F.conv2d(orig, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(adv, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(orig * orig, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(adv * adv, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(orig * adv, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # Mean over spatial and channel dims -> (B,)
    ssim_per_sample = ssim_map.mean(dim=[1, 2, 3])
    return ssim_per_sample


def compute_per_sample_lpips(
    orig: torch.Tensor, 
    adv: torch.Tensor, 
    lpips_fn: Optional[Any] = None
) -> Optional[torch.Tensor]:
    """
    Computes LPIPS distance per sample if lpips module is available.
    Expects input range [0, 1], converts to [-1, 1] internally for LPIPS.
    """
    if lpips_fn is None:
        return None
    with torch.no_grad():
        # Scale [0, 1] to [-1, 1]
        orig_norm = orig * 2.0 - 1.0
        adv_norm = adv * 2.0 - 1.0
        dist = lpips_fn(orig_norm, adv_norm) # (B, 1, 1, 1)
        return dist.view(-1)
