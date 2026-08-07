import os
import sys
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval

THIRD_PARTY_SPARSE_RS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/sparse_rs"))

class SparseRSOfficialAdapter:
    """
    Adapter wrapping official author implementation of Sparse-RS (fra31/sparse-rs, AAAI 2022).
    """
    def __init__(self, model: nn.Module, n_pixels: int = 15, n_queries: int = 1000, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.n_pixels = n_pixels
        self.n_queries = n_queries
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with scoped_sys_path(THIRD_PARTY_SPARSE_RS):
            from rs_attacks import RSAttack

            def predict_fn(inputs):
                with torch.no_grad():
                    return self.model(inputs)

            official_attacker = RSAttack(
                predict=predict_fn,
                norm='L0',
                n_queries=self.n_queries,
                eps=self.n_pixels,
                verbose=False,
                device=self.device
            )

            x = x.to(self.device)
            y = y.to(self.device)
            res = official_attacker.perturb(x, y)
            if isinstance(res, tuple):
                qr, adv_x = res
                self.last_steps = qr.cpu().numpy().tolist()
                return adv_x
            return res
