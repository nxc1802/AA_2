import torch
import torch.nn as nn
from src.attacks.optimization.pgd0 import PGD0Attack

class SparsePGDAttack(PGD0Attack):
    """Unified Sparse-PGD Attack (identical to PGD-0)."""
    def __init__(self, model, sparsity_budget=15, steps=25, alpha=4/255.0, device=None):
        super().__init__(model, k=sparsity_budget, steps=steps, alpha=alpha, device=device)
