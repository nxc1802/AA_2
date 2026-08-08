import os
import torch
import torch.nn as nn
from src.attacks.adapters.utils import scoped_sys_path
from src.core.utils import prepare_model_for_eval, get_best_device

THIRD_PARTY_SIA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/sparse_imperceivable_attacks"))

class CornerSearchOfficialAdapter:
    """
    Adapter wrapping official author implementation of CornerSearch (fra31/sparse-imperceivable-attacks, ICCV 2019).
    """
    def __init__(self, model: nn.Module, k: int = 15, max_pixels: int = None, max_iter: int = 1000, n_max: int = 100, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.k = max_pixels if max_pixels is not None else k
        self.max_iter = max_iter
        self.n_max = n_max
        self.device = device if device is not None else get_best_device()

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not os.path.exists(THIRD_PARTY_SIA):
            raise RuntimeError(f"Official CornerSearch dependency not found at '{THIRD_PARTY_SIA}'")
        with scoped_sys_path(THIRD_PARTY_SIA):
            try:
                from cornersearch_attacks_pt import CSattack
            except ImportError as e:
                raise RuntimeError(f"Failed to import official CornerSearch module: {e}")

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
            if hasattr(queries, "__len__"):
                self.last_queries = [int(q) for q in queries]
            else:
                self.last_queries = [int(queries)] * x.size(0)
            adv_tensor = torch.from_numpy(adv_np).permute(0, 3, 1, 2).to(self.device).float()
            return adv_tensor
