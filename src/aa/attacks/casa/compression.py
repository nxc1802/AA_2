import torch
import torch.nn as nn
from typing import Tuple

from aa.attacks.casa.objective import CoalitionObjective
from aa.attacks.casa.value_optimization import optimize_fixed_support


def drop_and_repair_support(
    model: nn.Module,
    objective: CoalitionObjective,
    orig_x: torch.Tensor,
    y: torch.Tensor,
    final_delta: torch.Tensor,
    final_support: torch.Tensor,
    k: int,
    repair_steps: int,
    alpha: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Prunes redundant active pixels on successful adversarial examples using Drop-and-Repair.
    Returns: (compressed_delta, compressed_support, fwd_evals, bwd_evals)
    """
    device = orig_x.device
    B, C, H, W = orig_x.shape
    HW = H * W
    b_idx = torch.arange(B, device=device)
    forward_evals = 0
    backward_evals = 0

    with torch.no_grad():
        x_adv_final = torch.clamp(orig_x + final_delta, 0.0, 1.0)
        logits_final = model(x_adv_final)
        forward_evals += 1
        succ = (logits_final.argmax(dim=1) != y)

    active_counts = final_support.view(B, HW).sum(dim=1)
    max_drop_passes = min(k, 6)

    for pass_idx in range(max_drop_passes):
        active_mask = (active_counts > 1) & succ
        if not active_mask.any():
            break

        with torch.enable_grad():
            x_curr = (orig_x + final_delta).detach().requires_grad_(True)
            loss_single = objective.compute_loss(model(x_curr), y).sum()
            forward_evals += 1
            backward_evals += 1
            loss_single.backward()
            g_curr = x_curr.grad if x_curr.grad is not None else torch.zeros_like(x_curr)

        redundancy = (g_curr * final_delta).sum(dim=1, keepdim=True)
        red_flat = redundancy.reshape(B, HW).clone()
        red_flat[~final_support.reshape(B, HW)] = float("inf")

        sorted_red_indices = red_flat.argsort(dim=1)
        any_pruned_in_pass = False
        max_rank_to_try = min(3, k)

        for rank_k in range(max_rank_to_try):
            i_star = sorted_red_indices[:, rank_k]

            test_supp_flat = final_support.reshape(B, HW).clone()
            test_supp_flat[b_idx, i_star] = False
            test_support = test_supp_flat.reshape(B, 1, H, W)

            test_delta = final_delta.clone()
            test_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0

            with torch.no_grad():
                x_test = torch.clamp(orig_x + test_delta, 0.0, 1.0)
                l_test = model(x_test)
                forward_evals += 1
                direct_succ = (l_test.argmax(dim=1) != y) & active_mask

            repair_mask = active_mask & (~direct_succ)
            repaired_delta = test_delta.clone()
            repair_succ = torch.zeros(B, dtype=torch.bool, device=device)

            if repair_mask.any():
                rep_delta, _, fwd, bwd = optimize_fixed_support(
                    model, objective, orig_x, y, test_support, test_delta,
                    num_steps=repair_steps, alpha=alpha
                )
                forward_evals += fwd
                backward_evals += bwd
                with torch.no_grad():
                    x_rep = torch.clamp(orig_x + rep_delta, 0.0, 1.0)
                    l_rep = model(x_rep)
                    forward_evals += 1
                    repair_succ = (l_rep.argmax(dim=1) != y) & repair_mask
                repaired_delta = rep_delta

            drop_succ = direct_succ | repair_succ
            if drop_succ.any():
                d_mask = drop_succ.reshape(B, 1, 1, 1)
                final_support = torch.where(d_mask, test_support, final_support)
                chosen_delta = torch.where(direct_succ.reshape(B, 1, 1, 1), test_delta, repaired_delta)
                final_delta = torch.where(d_mask, chosen_delta, final_delta)
                active_counts = final_support.reshape(B, HW).sum(dim=1)
                any_pruned_in_pass = True
                break

        if not any_pruned_in_pass:
            break

    return final_delta, final_support, forward_evals, backward_evals
