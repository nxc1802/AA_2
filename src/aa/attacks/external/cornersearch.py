import os
import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SIA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/sparse_imperceivable_attacks"))


class CornerSearch(Attack):
    def __init__(self, model: nn.Module, k: int = 16, max_iter: int = 1000, n_max: int = 100):
        self.model = model
        self.k = k
        self.max_iter = max_iter
        self.n_max = n_max

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        if not os.path.exists(THIRD_PARTY_SIA):
            raise RuntimeError(f"Official CornerSearch dependency not found at '{THIRD_PARTY_SIA}'")

        with scoped_sys_path(THIRD_PARTY_SIA):
            from cornersearch_attacks_pt import CSattack

            args = {
                'type_attack': 'L0',
                'n_iter': self.max_iter,
                'n_max': self.n_max,
                'epsilon': 0.0,
                'kappa': 0.0,
                'sparsity': self.k,
                'size_incr': 1
            }
            official_attacker = CSattack(self.model, args)

            x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
            y_np = y.detach().cpu().numpy()

            adv_np, _, queries, _ = official_attacker.perturb(x_np, y_np)
            total_queries = int(sum(queries)) if hasattr(queries, "__len__") else int(queries)
            adv_tensor = torch.from_numpy(adv_np).permute(0, 3, 1, 2).to(device).float()

            return AttackOutput(x_adv=adv_tensor, queries=total_queries)
