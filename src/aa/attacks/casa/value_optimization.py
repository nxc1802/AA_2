import torch
import torch.nn as nn
from typing import Tuple

from aa.attacks.casa.objective import CoalitionObjective


def optimize_fixed_support(
    model: nn.Module,
    objective: CoalitionObjective,
    x: torch.Tensor,
    y: torch.Tensor,
    support_mask: torch.Tensor,
    init_delta: torch.Tensor,
    num_steps: int,
    alpha: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Optimizes RGB perturbation delta on fixed support_mask (B, 1, H, W).
    Uses adaptive decaying step size to traverse box [0, 1] and settle accurately.
    Returns: (best_delta, best_margin, fwd_evals, bwd_evals)
    """
    B, C, H, W = x.shape
    fwd_evals = 0
    bwd_evals = 0

    curr_delta = (init_delta * support_mask).clone().detach()

    with torch.no_grad():
        x_adv_init = torch.clamp(x + curr_delta, 0.0, 1.0)
        logits_init = model(x_adv_init)
        fwd_evals += 1
        best_margin = objective.compute_margin(logits_init, y)
        best_delta = curr_delta.clone()

    for step in range(num_steps):
        curr_delta.requires_grad_(True)
        x_adv = torch.clamp(x + curr_delta * support_mask, 0.0, 1.0)
        logits = model(x_adv)
        loss = objective.compute_loss(logits, y).sum()
        fwd_evals += 1
        bwd_evals += 1

        model.zero_grad()
        loss.backward()

        grad = curr_delta.grad
        if grad is None:
            break

        with torch.no_grad():
            cur_alpha = alpha * (0.85 ** step)
            step_delta = curr_delta + cur_alpha * grad.sign() * support_mask
            new_delta = torch.clamp(x + step_delta, 0.0, 1.0) - x
            curr_delta = new_delta * support_mask

            logits_check = model(torch.clamp(x + curr_delta, 0.0, 1.0))
            fwd_evals += 1
            curr_margin = objective.compute_margin(logits_check, y)

            improved = curr_margin > best_margin
            if improved.any():
                best_margin = torch.where(improved, curr_margin, best_margin)
                imp_mask = improved.view(B, 1, 1, 1)
                best_delta = torch.where(imp_mask, curr_delta, best_delta)

    return best_delta, best_margin, fwd_evals, bwd_evals
