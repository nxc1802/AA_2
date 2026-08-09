import os
import torch
import torch.nn as nn
from typing import Optional
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SPARSEFOOL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/sparsefool"))


class SparseFool(Attack):
    def __init__(self, model: nn.Module, k: int = 250, steps: int = 20, max_iter: Optional[int] = None, lambda_val: float = 3.0):
        self.model = model
        self.k = k
        self.steps = max_iter if max_iter is not None else steps
        self.lambda_val = lambda_val

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = x.device

        with scoped_sys_path(THIRD_PARTY_SPARSEFOOL):
            from sparsefool import sparsefool

            x = x.to(device)
            y = y.to(device)
            B = x.size(0)
            x_adv = x.clone()

            total_loops = 0
            for b in range(B):
                x_single = x[b:b+1].to(device)
                fool_im, r, p_label, f_label, loops = sparsefool(
                    x_single, self.model, lb=0.0, ub=1.0, lambda_=self.lambda_val, max_iter=self.steps, device=device
                )
                x_adv[b] = fool_im[0].detach()
                total_loops += loops

            return AttackOutput(
                x_adv=x_adv,
                forward_evals=total_loops,
                backward_evals=total_loops
            )
