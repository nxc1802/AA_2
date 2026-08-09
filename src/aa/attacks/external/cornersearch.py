import os
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.scoped_path import scoped_sys_path

THIRD_PARTY_SIA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../third_party/sparse_imperceivable_attacks"))


class CornerSearch(Attack):
    def __init__(
        self,
        model: nn.Module,
        k: int = 64,
        max_k: Optional[int] = None,
        max_iter: int = 1000,
        n_max: int = 100,
        seed: Optional[int] = 42
    ):
        self.model = model
        self.k = max_k if max_k is not None else k
        self.max_k = self.k
        self.max_iter = max_iter
        self.n_max = n_max
        self.seed = seed

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        if not os.path.exists(THIRD_PARTY_SIA):
            raise RuntimeError(f"Official CornerSearch dependency not found at '{THIRD_PARTY_SIA}'")

        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        try:
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

                adv_np, pixels_changed, queries, upstream_success = official_attacker.perturb(x_np, y_np)
                total_queries = int(sum(queries)) if hasattr(queries, "__len__") else int(queries)
                adv_tensor = torch.from_numpy(adv_np).permute(0, 3, 1, 2).to(device).float()
        finally:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_state)

        # P0.4 — Strict validation post-attack
        # 1. Shape check
        if adv_tensor.shape != x.shape:
            raise RuntimeError(f"CornerSearch output shape {adv_tensor.shape} mismatch with input shape {x.shape}")

        # 2. Finite values check
        if not torch.isfinite(adv_tensor).all():
            raise RuntimeError("CornerSearch output contains non-finite values (NaN/Inf)")

        # 3. Bound check [0, 1]
        if (adv_tensor < 0.0).any() or (adv_tensor > 1.0).any():
            raise RuntimeError("CornerSearch output violates [0, 1] bounds")

        # 4. Spatial L0 <= Kmax check
        diff = adv_tensor - x
        aa2_l0 = (diff.abs().max(dim=1).values > 1e-5).sum(dim=(1, 2)).cpu()
        if (aa2_l0 > self.k).any():
            raise RuntimeError(
                f"CornerSearch output spatial L0 ({aa2_l0.max().item()}) exceeds Kmax ({self.k})"
            )

        # Upstream diagnostic cross-check
        pixels_changed_arr = np.array(pixels_changed, dtype=np.int64)
        upstream_l0_tensor = torch.from_numpy(pixels_changed_arr)
        l0_diff = (aa2_l0.long() - upstream_l0_tensor).abs()
        if (l0_diff > 1).any():
            print(
                f"[CornerSearch Warning] Discrepancy between AA2 L0 ({aa2_l0.tolist()}) "
                f"and upstream pixels_changed ({pixels_changed_arr.tolist()})",
                flush=True
            )

        metadata = {
            "upstream_pixels_changed": pixels_changed_arr.tolist(),
            "upstream_success": (
                upstream_success.tolist()
                if hasattr(upstream_success, "tolist")
                else list(upstream_success)
            ),
            "aa2_l0": aa2_l0.tolist(),
            "attack_seed": self.seed,
            "k_max": self.k,
        }

        return AttackOutput(x_adv=adv_tensor, queries=total_queries, metadata=metadata)

