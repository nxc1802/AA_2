from contextlib import redirect_stdout
import io
import os
import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SIA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/sparse_imperceivable_attacks"))


class PGD0(Attack):
    def __init__(self, model: nn.Module, k: int = 16, steps: int = 25, alpha: float = 4 / 255.0, verbose: bool = False):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.verbose = verbose

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
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

            if not self.verbose:
                with redirect_stdout(io.StringIO()):
                    adv_np, _ = official_attacker.perturb(x_np, y_np)
            else:
                adv_np, _ = official_attacker.perturb(x_np, y_np)

            adv_tensor = torch.from_numpy(adv_np).permute(0, 3, 1, 2).to(device).float()

            return AttackOutput(
                x_adv=adv_tensor,
                forward_evals=self.steps,
                backward_evals=self.steps
            )
