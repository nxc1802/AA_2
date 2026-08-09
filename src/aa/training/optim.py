import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR, _LRScheduler
from typing import Dict, Any, Tuple, Optional


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    Creates an optimizer based on configuration dict.
    Example config:
      name: sgd
      lr: 0.1
      momentum: 0.9
      weight_decay: 0.0005
      nesterov: false
    """
    opt_name = config.get("name", "sgd").lower()
    lr = float(config.get("lr", 0.1))
    weight_decay = float(config.get("weight_decay", 5e-4))

    if opt_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        nesterov = bool(config.get("nesterov", False))
        return SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    elif opt_name == "adam":
        betas = config.get("betas", (0.9, 0.999))
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas))
    elif opt_name == "adamw":
        betas = config.get("betas", (0.9, 0.999))
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas))
    else:
        raise ValueError(f"Unsupported optimizer name: '{opt_name}'")


def create_scheduler(optimizer: Optimizer, config: Dict[str, Any], total_epochs: int = 200) -> Optional[_LRScheduler]:
    """
    Creates a learning rate scheduler based on configuration dict.
    Example config:
      name: cosine
      min_lr: 0.0
    """
    sched_name = config.get("name", "cosine").lower()
    epochs = int(config.get("epochs", total_epochs))

    if sched_name in ["cosine", "cosine_annealing"]:
        min_lr = float(config.get("min_lr", 0.0))
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif sched_name in ["multistep", "multi_step"]:
        milestones = config.get("milestones", [100, 150])
        gamma = float(config.get("gamma", 0.1))
        return MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)
    elif sched_name == "step":
        step_size = int(config.get("step_size", 30))
        gamma = float(config.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name in ["none", "constant"]:
        return None
    else:
        raise ValueError(f"Unsupported scheduler name: '{sched_name}'")
