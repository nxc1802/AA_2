import os
import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SPARSE_RS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/sparse_rs"))


class SparseRS(Attack):
    def __init__(self, model: nn.Module, k: int = 16, n_queries: int = 1000):
        self.model = model
        self.k = k
        self.n_queries = n_queries

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        if not os.path.exists(THIRD_PARTY_SPARSE_RS):
            raise RuntimeError(f"Official Sparse-RS dependency not found at '{THIRD_PARTY_SPARSE_RS}'")

        with scoped_sys_path(THIRD_PARTY_SPARSE_RS):
            from rs_attacks import RSAttack

            def predict_fn(inputs):
                with torch.no_grad():
                    return self.model(inputs)

            official_attacker = RSAttack(
                predict=predict_fn,
                norm='L0',
                n_queries=self.n_queries,
                eps=self.k,
                verbose=False,
                device=device
            )

            x = x.to(device)
            y = y.to(device)
            res = official_attacker.perturb(x, y)
            if isinstance(res, tuple):
                qr, adv_x = res
                queries_count = int(qr.sum().item())
                return AttackOutput(x_adv=adv_x, queries=queries_count)
            return AttackOutput(x_adv=res, queries=self.n_queries)
