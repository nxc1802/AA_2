import os
import torch
import torch.nn as nn
from typing import Optional
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_GSE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/gse"))


class GSE(Attack):
    def __init__(self, model: nn.Module, k: int = 16, steps: int = 50, max_evals: Optional[int] = None):
        self.model = model
        self.k = k
        self.steps = max_evals if max_evals is not None else steps

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        with scoped_sys_path(THIRD_PARTY_GSE):
            from attacks import GSEAttack

            official_attacker = GSEAttack(
                model=self.model,
                img_range=(0, 1),
                iters=self.steps
            )

            x = x.to(device)
            y = y.to(device)
            adv_x = official_attacker(x, y)

            return AttackOutput(
                x_adv=adv_x,
                forward_evals=self.steps,
                backward_evals=self.steps
            )
