import os
import torch
import torch.nn as nn
from typing import Optional

from aa.attacks.base import Attack, AttackOutput
from aa.attacks.external.spgd import SparsePGD
from aa.attacks.external.sparse_rs import SparseRS
from aa.metrics import compute_spatial_l0


class SparseAutoAttack(Attack):
    """
    Standardized Sparse-AutoAttack (sAA) ensemble adapter.
    Sequentially applies complementary sparse attacks on unfooled samples:
      1. Sparse-PGD (Projected gradient)
      2. Sparse-PGD (Unprojected gradient)
      3. Sparse-RS (Derivative-free random search, optional / for unfooled)
    """
    def __init__(
        self,
        model: nn.Module,
        k: int = 16,
        spgd_steps: int = 100,
        alpha: float = 0.25,
        include_rs: bool = True,
        rs_queries: int = 1000,
    ):
        self.model = model
        self.k = k
        self.spgd_steps = spgd_steps
        self.alpha = alpha
        self.include_rs = include_rs
        self.rs_queries = rs_queries

        self.spgd_proj = SparsePGD(
            model=model,
            k=k,
            steps=spgd_steps,
            alpha=alpha,
            unprojected_gradient=False
        )
        self.spgd_unproj = SparsePGD(
            model=model,
            k=k,
            steps=spgd_steps,
            alpha=alpha,
            unprojected_gradient=True
        )
        if self.include_rs:
            self.sparse_rs = SparseRS(
                model=model,
                k=k,
                n_queries=rs_queries
            )
        else:
            self.sparse_rs = None

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        B = x.shape[0]
        total_fwd = 0
        total_bwd = 0
        total_queries = 0

        # Stage 1: Sparse-PGD (Projected)
        out_proj = self.spgd_proj.attack(x, y)
        total_fwd += getattr(out_proj, "forward_evals", 0)
        total_bwd += getattr(out_proj, "backward_evals", 0)
        total_queries += getattr(out_proj, "queries", 0)

        best_x_adv = out_proj.x_adv.clone()
        with torch.no_grad():
            preds = self.model(best_x_adv).argmax(dim=1)
            fooled = preds != y

        # If all samples fooled, return early
        if fooled.all():
            return AttackOutput(
                x_adv=best_x_adv,
                forward_evals=total_fwd,
                backward_evals=total_bwd,
                queries=total_queries,
                metadata={"k": self.k, "stopped_at": "spgd_projected"}
            )

        # Stage 2: Sparse-PGD (Unprojected) on remaining unfooled samples
        unfooled_idx = (~fooled).nonzero(as_tuple=True)[0]
        x_unfooled = x[unfooled_idx]
        y_unfooled = y[unfooled_idx]

        out_unproj = self.spgd_unproj.attack(x_unfooled, y_unfooled)
        total_fwd += getattr(out_unproj, "forward_evals", 0)
        total_bwd += getattr(out_unproj, "backward_evals", 0)
        total_queries += getattr(out_unproj, "queries", 0)

        with torch.no_grad():
            preds_unproj = self.model(out_unproj.x_adv).argmax(dim=1)
            new_fooled = preds_unproj != y_unfooled

        # Update best adversarial examples for newly fooled samples
        best_x_adv[unfooled_idx[new_fooled]] = out_unproj.x_adv[new_fooled]
        fooled[unfooled_idx[new_fooled]] = True

        if fooled.all() or not self.include_rs or self.sparse_rs is None:
            return AttackOutput(
                x_adv=best_x_adv,
                forward_evals=total_fwd,
                backward_evals=total_bwd,
                queries=total_queries,
                metadata={"k": self.k, "stopped_at": "spgd_unprojected"}
            )

        # Stage 3: Sparse-RS on remaining stubborn samples
        still_unfooled = (~fooled).nonzero(as_tuple=True)[0]
        if len(still_unfooled) > 0:
            x_stubborn = x[still_unfooled]
            y_stubborn = y[still_unfooled]

            out_rs = self.sparse_rs.attack(x_stubborn, y_stubborn)
            total_fwd += getattr(out_rs, "forward_evals", 0)
            total_bwd += getattr(out_rs, "backward_evals", 0)
            total_queries += getattr(out_rs, "queries", 0)

            with torch.no_grad():
                preds_rs = self.model(out_rs.x_adv).argmax(dim=1)
                rs_fooled = preds_rs != y_stubborn

            best_x_adv[still_unfooled[rs_fooled]] = out_rs.x_adv[rs_fooled]

        return AttackOutput(
            x_adv=best_x_adv,
            forward_evals=total_fwd,
            backward_evals=total_bwd,
            queries=total_queries,
            metadata={"k": self.k, "stopped_at": "sparse_rs"}
        )
