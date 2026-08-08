import os
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval, get_best_device

THIRD_PARTY_GSE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/gse"))

class GSEOfficialAdapter:
    """
    Adapter wrapping official author implementation of GSE Attack (wagnermoritz/GSE, ICLR 2025).
    Features 1/2-quasinorm proximal optimization and projected Nesterov acceleration.
    """
    def __init__(self, model: nn.Module, k: int = 15, group_size: int = 4, max_groups: int = None, steps: int = 50, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = k
        self.steps = steps
        self.device = device if device is not None else get_best_device()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with scoped_sys_path(THIRD_PARTY_GSE):
            from attacks import GSEAttack

            official_attacker = GSEAttack(
                model=self.model,
                img_range=(0, 1),
                iters=self.steps
            )

            x = x.to(self.device)
            y = y.to(self.device)
            adv_x = official_attacker(x, y)
            return adv_x
