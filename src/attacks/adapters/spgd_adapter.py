import os
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval, get_best_device

THIRD_PARTY_SPGD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/spgd/adversarial_training"))

class SparsePGDOfficialAdapter:
    """
    Adapter wrapping official author implementation of sPGD (CityU-MLO/sPGD, ICML 2024).
    Uses separated magnitude and sparsity mask optimization.
    """
    def __init__(self, model: nn.Module, sparsity_budget: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = sparsity_budget
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with scoped_sys_path(THIRD_PARTY_SPGD):
            from spgd import SparsePGD

            official_attacker = SparsePGD(
                model=self.model,
                epsilon=1.0,
                k=self.k,
                t=self.steps,
                random_start=True,
                attack_mode="pixel"
            )

            x = x.to(self.device)
            y = y.to(self.device)
            adv_x = official_attacker.perturb(x, y)
            if isinstance(adv_x, tuple):
                adv_x = adv_x[0]
            return adv_x
