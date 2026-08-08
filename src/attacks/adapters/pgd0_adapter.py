import os
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval, get_best_device

THIRD_PARTY_SIA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/sparse_imperceivable_attacks"))

class PGD0OfficialAdapter:
    """
    Adapter wrapping official author implementation of PGD0 (fra31/sparse-imperceivable-attacks, ICCV 2019).
    """
    def __init__(self, model: nn.Module, k: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else get_best_device()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with scoped_sys_path(THIRD_PARTY_SIA):
            from pgd_attacks_pt import PGDattack

            args = {
                'type_attack': 'L0',
                'num_steps': self.steps,
                'step_size': self.alpha,
                'n_restarts': 1,
                'epsilon': 0.0,
                'kappa': 0.0,
                'sparsity': self.k
            }
            official_attacker = PGDattack(self.model, args)

            x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
            y_np = y.detach().cpu().numpy()

            adv_np, _ = official_attacker.perturb(x_np, y_np)
            adv_tensor = torch.from_numpy(adv_np).permute(0, 3, 1, 2).to(self.device).float()
            return adv_tensor
