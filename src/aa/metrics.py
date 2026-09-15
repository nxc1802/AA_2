import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, NamedTuple


class BatchMetrics(NamedTuple):
    clean_correct: torch.Tensor
    adv_correct: torch.Tensor
    success: torch.Tensor
    l0: torch.Tensor
    l2: torch.Tensor
    linf: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor
    lpips: Optional[torch.Tensor]


def compute_spatial_l0(delta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Computes spatial L0 norm per sample in batch (modified pixels)."""
    channel_max = delta.abs().max(dim=1).values
    l0_per_sample = (channel_max > eps).flatten(1).sum(dim=1)
    return l0_per_sample


def exact_spatial_topk_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    """Constructs boolean spatial mask selecting exactly k pixels per sample."""
    if score.dim() == 4:
        score_spatial = score.squeeze(1)
    else:
        score_spatial = score

    B, H, W = score_spatial.shape
    num_pixels = H * W
    if k <= 0:
        return torch.zeros((B, 1, H, W), dtype=torch.bool, device=score.device)
    k_bounded = min(k, num_pixels)

    flat_score = score_spatial.flatten(1)
    indices = flat_score.topk(k_bounded, dim=1).indices

    flat_mask = torch.zeros_like(flat_score, dtype=torch.bool)
    flat_mask.scatter_(1, indices, True)

    mask = flat_mask.view(B, 1, H, W)
    return mask


def project_l0(delta: torch.Tensor, k: int) -> torch.Tensor:
    """Projects perturbation delta onto spatial L0-ball of radius k."""
    spatial_mag = torch.norm(delta, p=2, dim=1, keepdim=True)
    mask = exact_spatial_topk_mask(spatial_mag, k)
    return delta * mask.to(delta.dtype)


def compute_per_sample_psnr(orig: torch.Tensor, adv: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Computes PSNR per sample in dB."""
    mse = torch.mean((orig - adv) ** 2, dim=[1, 2, 3])
    mse_clamped = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse_clamped)
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
    """Computes SSIM per sample using Gaussian window."""
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
    return ssim_map.mean(dim=[1, 2, 3])


def compute_per_sample_lpips(orig: torch.Tensor, adv: torch.Tensor, lpips_fn: Optional[Any] = None) -> Optional[torch.Tensor]:
    """Computes LPIPS distance per sample if lpips_fn is provided."""
    if lpips_fn is None:
        return None
    with torch.no_grad():
        orig_norm = orig * 2.0 - 1.0
        adv_norm = adv * 2.0 - 1.0
        dist = lpips_fn(orig_norm, adv_norm)
        return dist.view(-1)


def evaluate_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    output,
    lpips_fn: Optional[Any] = None
) -> BatchMetrics:
    """Evaluates metrics on a batch of adversarial examples given model, x, y, output."""
    x_adv = output.x_adv
    delta = x_adv - x

    with torch.no_grad():
        if hasattr(model, "evaluate_defended"):
            clean_pred = model.evaluate_defended(x).argmax(dim=1)
            adv_pred = model.evaluate_defended(x_adv).argmax(dim=1)
        else:
            clean_pred = model(x).argmax(dim=1)
            adv_pred = model(x_adv).argmax(dim=1)

    clean_correct = clean_pred.eq(y)
    adv_correct = adv_pred.eq(y)
    success = clean_correct & adv_pred.ne(y)

    l0 = compute_spatial_l0(delta)
    l2 = delta.flatten(1).norm(p=2, dim=1)
    linf = delta.flatten(1).abs().max(dim=1).values
    psnr = compute_per_sample_psnr(x, x_adv)
    ssim = compute_per_sample_ssim(x, x_adv)
    lpips = compute_per_sample_lpips(x, x_adv, lpips_fn=lpips_fn)

    return BatchMetrics(
        clean_correct=clean_correct,
        adv_correct=adv_correct,
        success=success,
        l0=l0,
        l2=l2,
        linf=linf,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips
    )


def compute_distortion_metrics(
    l0_per: torch.Tensor,
    l2_per: torch.Tensor,
    linf_per: torch.Tensor,
    psnr_per: torch.Tensor,
    ssim_per: torch.Tensor,
    lpips_per: Optional[torch.Tensor],
    success_mask: torch.Tensor
) -> Dict[str, Any]:
    """Aggregates distortion metrics over all samples and successful samples.

    Reports mean, std, median, Q1 (25th percentile), Q3 (75th percentile) for L0
    on successful samples, providing richer summary statistics for paper tables.
    """
    total_count = l0_per.numel()
    succ_count = success_mask.sum().item()

    metrics = {
        "all_l0_mean": l0_per.float().mean().item() if total_count > 0 else 0.0,
        "all_l2_mean": l2_per.float().mean().item() if total_count > 0 else 0.0,
        "all_linf_mean": linf_per.float().mean().item() if total_count > 0 else 0.0,
        "succ_count": succ_count,
        "total_count": total_count,
    }

    if succ_count > 0:
        succ_l0 = l0_per[success_mask].float()
        succ_l2 = l2_per[success_mask].float()
        succ_linf = linf_per[success_mask].float()
        succ_psnr = psnr_per[success_mask].float()
        succ_ssim = ssim_per[success_mask].float()

        metrics["succ_l0_mean"] = succ_l0.mean().item()
        metrics["succ_l0_std"] = succ_l0.std().item() if succ_l0.numel() > 1 else 0.0
        metrics["succ_l0_median"] = succ_l0.median().item()
        metrics["succ_l0_q25"] = succ_l0.quantile(0.25).item()
        metrics["succ_l0_q75"] = succ_l0.quantile(0.75).item()
        metrics["succ_l2_mean"] = succ_l2.mean().item()
        metrics["succ_linf_mean"] = succ_linf.mean().item()
        metrics["succ_psnr_mean"] = succ_psnr.mean().item()
        metrics["succ_ssim_mean"] = succ_ssim.mean().item()
        metrics["succ_lpips_mean"] = lpips_per[success_mask].mean().item() if lpips_per is not None else None
    else:
        for key in [
            "succ_l0_mean", "succ_l0_std", "succ_l0_median",
            "succ_l0_q25", "succ_l0_q75",
            "succ_l2_mean", "succ_linf_mean",
            "succ_psnr_mean", "succ_ssim_mean", "succ_lpips_mean",
        ]:
            metrics[key] = None

    return metrics


import math


def compute_wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Computes Wilson score confidence interval for binomial proportion (ASR).
    Essential for statistical significance claims in adversarial robustness papers.
    """
    if total <= 0:
        return {"p": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "margin": 0.0}

    # For 95% confidence, z = 1.95996
    z_map = {0.90: 1.64485, 0.95: 1.95996, 0.99: 2.57583}
    z = z_map.get(confidence, 1.95996)

    p_hat = successes / total
    z2 = z * z
    n = total

    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * (n ** 2)))) / denominator

    ci_lower = max(0.0, center - spread) * 100.0
    ci_upper = min(1.0, center + spread) * 100.0
    p_pct = p_hat * 100.0

    return {
        "p": round(p_pct, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "margin": round((ci_upper - ci_lower) / 2.0, 4)
    }


def compute_relative_and_absolute_gain(asr_proposed: float, asr_baseline: float) -> Dict[str, float]:
    """Computes absolute and relative improvement between proposed attack and baseline."""
    abs_gain = asr_proposed - asr_baseline
    rel_gain = (abs_gain / asr_baseline * 100.0) if asr_baseline > 1e-6 else float("inf")
    return {
        "abs_gain": round(abs_gain, 4),
        "rel_gain": round(rel_gain, 2)
    }

