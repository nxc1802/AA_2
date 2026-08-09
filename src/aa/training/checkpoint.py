import os
import random
import subprocess
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from aa.utils import compute_file_sha256


def get_git_commit() -> str:
    """Returns current git commit hash if inside a git repo."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_rng_states() -> Dict[str, Any]:
    """Captures all random number generator states for reproducibility."""
    states = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restores all random number generator states."""
    if "torch" in states and states["torch"] is not None:
        torch.set_rng_state(states["torch"])
    if "python" in states and states["python"] is not None:
        random.setstate(states["python"])
    if "numpy" in states and states["numpy"] is not None:
        np.random.set_state(states["numpy"])
    if "cuda" in states and states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


class CheckpointManager:
    """
    Manages saving and loading model checkpoints with strict metadata tracking
    and exact RNG reproducibility support.
    """
    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        dataset_name: str = "cifar10",
        architecture: str = "resnet18",
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
        monitor: str = "val_accuracy",
        mode: str = "max"
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.experiment_dir = os.path.join(self.checkpoint_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.architecture = architecture
        self.seed = seed
        self.config = config or {}
        self.monitor = monitor
        self.mode = mode.lower()

        if self.mode == "max":
            self.best_metric = -float("inf")
        else:
            self.best_metric = float("inf")
        self.best_epoch = 0

    def is_better(self, metric: float) -> bool:
        if self.mode == "max":
            return metric > self.best_metric
        else:
            return metric < self.best_metric

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        current_metric: float,
        is_best: bool = False
    ) -> Tuple[str, str]:
        """
        Saves checkpoint payload to disk (`last.pth` and optionally `best.pth`).
        Returns tuple of (last_path, last_sha256).
        """
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": self.best_metric,
            "current_metric": current_metric,
            "dataset": self.dataset_name,
            "architecture": self.architecture,
            "seed": self.seed,
            "config": self.config,
            "git_commit": get_git_commit(),
            "rng_states": get_rng_states(),
        }

        last_path = os.path.join(self.experiment_dir, "last.pth")
        torch.save(payload, last_path)
        last_sha256 = compute_file_sha256(last_path)

        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch
            payload["best_metric"] = self.best_metric
            best_path = os.path.join(self.experiment_dir, "best.pth")
            torch.save(payload, best_path)

        return last_path, last_sha256

    def load_resume(
        self,
        resume_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> int:
        """
        Loads a checkpoint to resume training.
        Restores model weights, optimizer state, scheduler state, RNG states, and returns starting epoch.
        """
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint file not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device or "cpu", weights_only=False)

        # Load weights and optimizer
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "best_metric" in checkpoint:
            self.best_metric = checkpoint["best_metric"]

        if "rng_states" in checkpoint and checkpoint["rng_states"] is not None:
            set_rng_states(checkpoint["rng_states"])

        start_epoch = checkpoint.get("epoch", 0)
        return start_epoch
