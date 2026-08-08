import random
import numpy as np
import torch
import torch.nn as nn

def get_best_device() -> torch.device:
    """
    Returns the best available hardware accelerator (CUDA for NVIDIA, MPS for Apple Silicon, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def synchronize_device(device: torch.device = None) -> None:
    """
    Synchronizes device operations for accurate benchmarking timing (supports CUDA and MPS).
    """
    if device is None:
        device = get_best_device()
    dev_type = device.type if hasattr(device, "type") else str(device)
    if dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_device_name(device: torch.device = None) -> str:
    """
    Returns a human-readable device identifier.
    """
    if device is None:
        device = get_best_device()
    dev_type = device.type if hasattr(device, "type") else str(device)
    if dev_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif dev_type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def prepare_model_for_eval(model: nn.Module, device: torch.device = None) -> nn.Module:
    """
    Puts model strictly into eval mode, moves it to target device, and freezes all model parameters.
    Prevents BatchNorm running statistics updates during adversarial attacks.
    Inputs can still require_grad for backward pass.
    """
    if device is None:
        device = get_best_device()
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 checksum of a file in 1MB chunks."""
    import hashlib
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_reproducibility_info(repo_dir: str = None) -> dict:
    """
    Returns current Git commit SHA and working tree dirty status for reproducibility metadata.
    """
    import subprocess
    import os

    if repo_dir is None:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        is_dirty = len(status_output) > 0
    except Exception:
        commit_sha = "unknown"
        is_dirty = True

    return {
        "git_commit": commit_sha,
        "git_dirty": is_dirty
    }
