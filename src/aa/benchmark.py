import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional

from aa.utils import get_best_device, synchronize_device
from aa.metrics import evaluate_batch, compute_distortion_metrics, BatchMetrics


def evaluate_attack(
    model: nn.Module,
    attack: Any,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    lpips_fn: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluates an adversarial attack over a DataLoader using a single generic loop.
    
    Computes:
    - Clean accuracy
    - Robust accuracy
    - Conditional ASR (clean-correct conditioned)
    - Distortion metrics (L0, L2, Linf, PSNR, SSIM, LPIPS)
    - Forward/backward/query counts & runtime
    """
    if device is None:
        device = get_best_device()

    model.eval()

    all_clean_correct = []
    all_success = []
    all_l0 = []
    all_l2 = []
    all_linf = []
    all_psnr = []
    all_ssim = []

    total_forward = 0
    total_backward = 0
    total_queries = 0

    synchronize_device(device)
    start_time = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        output = attack.attack(x, y)
        total_forward += getattr(output, "forward_evals", 0)
        total_backward += getattr(output, "backward_evals", 0)
        total_queries += getattr(output, "queries", 0)

        batch_m: BatchMetrics = evaluate_batch(model, x, y, output)

        all_clean_correct.append(batch_m.clean_correct.cpu())
        all_success.append(batch_m.success.cpu())
        all_l0.append(batch_m.l0.cpu())
        all_l2.append(batch_m.l2.cpu())
        all_linf.append(batch_m.linf.cpu())
        all_psnr.append(batch_m.psnr.cpu())
        all_ssim.append(batch_m.ssim.cpu())

    synchronize_device(device)
    elapsed_time = time.time() - start_time

    clean_corr_cat = torch.cat(all_clean_correct)
    success_cat = torch.cat(all_success)
    l0_cat = torch.cat(all_l0)
    l2_cat = torch.cat(all_l2)
    linf_cat = torch.cat(all_linf)
    psnr_cat = torch.cat(all_psnr)
    ssim_cat = torch.cat(all_ssim)

    total_samples = clean_corr_cat.numel()
    clean_correct_count = clean_corr_cat.sum().item()
    succ_count = success_cat.sum().item()

    clean_acc = 100.0 * clean_correct_count / total_samples if total_samples > 0 else 0.0
    robust_acc = 100.0 * (clean_correct_count - succ_count) / total_samples if total_samples > 0 else 0.0
    asr = 100.0 * succ_count / clean_correct_count if clean_correct_count > 0 else 0.0

    distortion_m = compute_distortion_metrics(
        l0_per=l0_cat,
        l2_per=l2_cat,
        linf_per=linf_cat,
        psnr_per=psnr_cat,
        ssim_per=ssim_cat,
        lpips_per=None,
        success_mask=success_cat
    )

    return {
        "total_samples": total_samples,
        "clean_correct_count": clean_correct_count,
        "success_count": succ_count,
        "clean_accuracy": clean_acc,
        "robust_accuracy": robust_acc,
        "asr": asr,
        "runtime_seconds": elapsed_time,
        "total_forward_evals": total_forward,
        "total_backward_evals": total_backward,
        "total_queries": total_queries,
        "metrics": distortion_m
    }
