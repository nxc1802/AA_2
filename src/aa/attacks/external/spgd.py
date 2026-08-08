import os
import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SPGD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/spgd/adversarial_training"))


class SparsePGD(Attack):
    def __init__(self, model: nn.Module, k: int = 16, steps: int = 25, alpha: float = 4 / 255.0):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        with scoped_sys_path(THIRD_PARTY_SPGD):
            from spgd import SparsePGD as SPGDImpl

            official_attacker = SPGDImpl(
                model=self.model,
                epsilon=1.0,
                k=self.k,
                t=self.steps,
                random_start=True,
                attack_mode="pixel"
            )

            x = x.to(device)
            y = y.to(device)
            res = official_attacker.perturb(x, y)
            adv_x = res[0] if isinstance(res, tuple) else res

            return AttackOutput(
                x_adv=adv_x,
                forward_evals=self.steps,
                backward_evals=self.steps
            )
