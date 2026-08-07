import random
import numpy as np
import torch
import torch.nn as nn

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


def prepare_model_for_eval(model: nn.Module, device: torch.device = None) -> nn.Module:
    """
    Puts model strictly into eval mode and freezes all model parameters.
    Prevents BatchNorm running statistics updates during adversarial attacks.
    Inputs can still require_grad for backward pass.
    """
    if device is not None:
        model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
