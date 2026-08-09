import random
import numpy as np
import torch
import torch.nn as nn
import hashlib
import os
import subprocess


def get_best_device() -> torch.device:
    """Returns CUDA, MPS (Apple Silicon), or CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def synchronize_device(device: torch.device = None) -> None:
    """Synchronizes device for accurate execution timing."""
    if device is None:
        device = get_best_device()
    dev_type = device.type if hasattr(device, "type") else str(device)
    if dev_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_device_name(device: torch.device = None) -> str:
    """Returns human-readable device name."""
    if device is None:
        device = get_best_device()
    dev_type = device.type if hasattr(device, "type") else str(device)
    if dev_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif dev_type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across python, numpy, and PyTorch."""
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
    """Freezes model parameters and switches to eval mode on specified device."""
    if device is None:
        device = get_best_device()
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class CountingModel(nn.Module):
    """Wraps PyTorch nn.Module to track exact forward calls and samples evaluated."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.forward_calls = 0
        self.samples_evaluated = 0

    def reset_counters(self):
        self.forward_calls = 0
        self.samples_evaluated = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.samples_evaluated += x.size(0)
        return self.model(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_reproducibility_info(repo_dir: str = None) -> dict:
    """Returns current Git commit hash, dirty status, and environment provenance."""
    import sys
    import platform

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
        "git_dirty": is_dirty,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "device_name": get_device_name()
    }

