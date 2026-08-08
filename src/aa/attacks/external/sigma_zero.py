import os
import sys
import types
import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SIGMA_ZERO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/sigma_zero"))


def ensure_adv_lib_shim():
    if 'adv_lib' not in sys.modules:
        adv_lib = types.ModuleType('adv_lib')
        utils = types.ModuleType('adv_lib.utils')
        losses = types.ModuleType('adv_lib.utils.losses')

        def difference_of_logits(logits, labels):
            batch_size = logits.shape[0]
            labels_logits = logits[torch.arange(batch_size), labels]
            other_logits = logits.clone()
            other_logits[torch.arange(batch_size), labels] = -float('inf')
            max_other_logits = other_logits.max(dim=1).values
            return labels_logits - max_other_logits

        losses.difference_of_logits = difference_of_logits
        utils.losses = losses
        adv_lib.utils = utils
        sys.modules['adv_lib'] = adv_lib
        sys.modules['adv_lib.utils'] = utils
        sys.modules['adv_lib.utils.losses'] = losses


class SigmaZero(Attack):
    def __init__(self, model: nn.Module, k: int = 16, steps: int = 50):
        self.model = model
        self.k = k
        self.steps = steps

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        if not os.path.exists(THIRD_PARTY_SIGMA_ZERO):
            raise RuntimeError(f"Official SigmaZero dependency not found at '{THIRD_PARTY_SIGMA_ZERO}'")

        ensure_adv_lib_shim()

        with scoped_sys_path(THIRD_PARTY_SIGMA_ZERO):
            from sigma_zero_attack import sigma_zero

            x = x.to(device)
            y = y.to(device)
            adv_x = sigma_zero(self.model, x, y, steps=self.steps)

            return AttackOutput(
                x_adv=adv_x,
                forward_evals=self.steps,
                backward_evals=self.steps
            )
