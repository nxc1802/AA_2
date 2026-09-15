import os
import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SPGD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/spgd/adversarial_training"))


class SparsePGD(Attack):
    """
    Adapter for official Sparse-PGD (sPGD) implementation (Croce & Hein, CityU-MLO).
    Supports projected and unprojected gradient variants, custom steps, and learning rates.
    """
    def __init__(
        self,
        model: nn.Module,
        k: int = 16,
        steps: int = 100,
        alpha: float = 0.25,
        beta: float = 0.25,
        patience: int = 3,
        unprojected_gradient: bool = False,
        random_start: bool = True,
        early_stop: bool = True,
        classes: int = 10
    ):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.patience = patience
        self.unprojected_gradient = unprojected_gradient
        self.random_start = random_start
        self.early_stop = early_stop
        self.classes = classes

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        with scoped_sys_path(THIRD_PARTY_SPGD):
            from spgd import SparsePGD as SPGDImpl

            official_attacker = SPGDImpl(
                model=self.model,
                epsilon=1.0,
                k=self.k,
                t=self.steps,
                alpha=self.alpha,
                beta=self.beta,
                patience=self.patience,
                unprojected_gradient=self.unprojected_gradient,
                random_start=self.random_start,
                early_stop=self.early_stop,
                classes=self.classes,
                attack_mode="pixel",
                verbose=False
            )

            x = x.to(device)
            y = y.to(device)
            res = official_attacker.perturb(x, y)
            adv_x = res[0] if isinstance(res, tuple) else res

            return AttackOutput(
                x_adv=adv_x,
                forward_evals=self.steps,
                backward_evals=self.steps,
                queries=self.steps,
                metadata={
                    "k": self.k,
                    "steps": self.steps,
                    "unprojected_gradient": self.unprojected_gradient,
                    "alpha": self.alpha
                }
            )
