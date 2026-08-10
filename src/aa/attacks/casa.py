import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from aa.attacks.base import Attack, AttackOutput
from aa.metrics import compute_spatial_l0, exact_spatial_topk_mask, project_l0


class CoalitionSparseAttack(Attack):
    """
    CASA — Coalition-Aware Sparse Adversarial Attack (Proposed V2).

    Key Innovations:
    1. Sparse support is evaluated as a coalition: candidate pixel utility depends on current support,
       measured via conditional gain: Delta(j | S) = F(S U {j}) - F(S).
    2. Candidate screening via Box-Aware Potential Gain A_i at clean image.
    3. Fixed-support RGB optimization: L0 constraint is structurally guaranteed without L0 projection.
    4. Redundancy-aware support exchange (1-out / 1-in swap) with exact accept/reject criterion.
    5. Drop-and-Repair support minimization to prune successful adversarial images to minimal L0.
    """
    def __init__(
        self,
        model: nn.Module,
        k: int = 16,
        steps: int = 20,
        inner_steps: int = 10,
        repair_steps: int = 5,
        alpha: float = 4 / 255.0,
        candidate_pool_size: Optional[int] = None,
        loss_fn: str = "margin",
        drop_and_repair: bool = True,
        pair_exploration: bool = True,
        pair_search_every: int = 5,
    ):
        self.model = model
        self.k = k
        self.steps = steps
        self.inner_steps = inner_steps
        self.repair_steps = repair_steps
        self.alpha = alpha
        self.candidate_pool_size = candidate_pool_size
        self.loss_fn = loss_fn.lower()
        self.drop_and_repair = drop_and_repair
        self.pair_exploration = pair_exploration
        self.pair_search_every = pair_search_every

    def _compute_margin(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes attack margin: J(x, y) = max_{c != y} z_c(x) - z_y(x).
        J > 0 implies misclassification.
        """
        B, C = logits.shape
        one_hot = F.one_hot(y, num_classes=C).bool()
        z_y = logits[one_hot]
        
        # Mask out true label logits with -inf to find max incorrect class logit
        logits_other = logits.clone()
        logits_other[one_hot] = float("-inf")
        z_other = logits_other.max(dim=1)[0]
        
        return z_other - z_y

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes loss to maximize."""
        if self.loss_fn == "margin":
            return self._compute_margin(logits, y)
        elif self.loss_fn == "ce":
            return F.cross_entropy(logits, y, reduction="none")
        else:
            raise ValueError(f"Unknown loss_fn '{self.loss_fn}'. Options: ['margin', 'ce']")

    def _compute_box_aware_gain(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Computes linear potential gain in [0, 1]^3 box for each spatial pixel:
        A_i = sum_c |g_{ic}| * (1 - x_{ic} if g_{ic} > 0 else x_{ic})
        Shape: (B, 1, H, W)
        """
        pos_gain = F.relu(grad) * (1.0 - x)
        neg_gain = F.relu(-grad) * x
        gain_rgb = pos_gain + neg_gain
        return gain_rgb.sum(dim=1, keepdim=True)  # (B, 1, H, W)

    def _optimize_fixed_support(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        support_mask: torch.Tensor,
        init_delta: torch.Tensor,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Optimizes RGB perturbation delta on fixed support_mask (B, 1, H, W).
        Returns: (best_delta, best_margin, fwd_evals, bwd_evals)
        """
        device = x.device
        B, C, H, W = x.shape
        fwd_evals = 0
        bwd_evals = 0

        curr_delta = (init_delta * support_mask).clone().detach()

        with torch.no_grad():
            x_adv_init = torch.clamp(x + curr_delta, 0.0, 1.0)
            logits_init = self.model(x_adv_init)
            fwd_evals += 1
            best_margin = self._compute_margin(logits_init, y)
            best_delta = curr_delta.clone()

        for step in range(num_steps):
            curr_delta.requires_grad_(True)
            x_adv = torch.clamp(x + curr_delta * support_mask, 0.0, 1.0)
            logits = self.model(x_adv)
            loss = self._compute_loss(logits, y).sum()
            fwd_evals += 1
            bwd_evals += 1

            self.model.zero_grad()
            loss.backward()

            grad = curr_delta.grad
            if grad is None:
                break

            with torch.no_grad():
                # Projected gradient ascent on RGB values inside support
                step_delta = curr_delta + self.alpha * grad.sign() * support_mask
                # Clip x + step_delta to [0, 1] box
                new_delta = torch.clamp(x + step_delta, 0.0, 1.0) - x
                curr_delta = new_delta * support_mask

                # Check margin improvement
                logits_check = self.model(torch.clamp(x + curr_delta, 0.0, 1.0))
                fwd_evals += 1
                curr_margin = self._compute_margin(logits_check, y)

                improved = curr_margin > best_margin
                if improved.any():
                    best_margin = torch.where(improved, curr_margin, best_margin)
                    imp_mask = improved.view(B, 1, 1, 1)
                    best_delta = torch.where(imp_mask, curr_delta, best_delta)

        return best_delta, best_margin, fwd_evals, bwd_evals

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        B, C, H, W = x.shape
        HW = H * W
        orig_x = x.clone().detach()
        y = y.clone().detach()

        forward_evals = 0
        backward_evals = 0

        # Candidate pool size M
        M = self.candidate_pool_size
        if M is None:
            M = min(64, max(16, 4 * self.k))
        M = min(M, HW)
        effective_k = min(self.k, HW)

        # -------------------------------------------------------------
        # Stage 1: Candidate Generation (Box-Aware Potential Gain at clean x)
        # -------------------------------------------------------------
        x_clean = orig_x.clone().requires_grad_(True)
        logits_clean = self.model(x_clean)
        forward_evals += 1
        loss_clean = self._compute_loss(logits_clean, y).sum()
        self.model.zero_grad()
        loss_clean.backward()
        backward_evals += 1

        grad_clean = x_clean.grad if x_clean.grad is not None else torch.zeros_like(x_clean)
        box_gain_clean = self._compute_box_aware_gain(orig_x, grad_clean)  # (B, 1, H, W)

        # Top-M Candidate Pool per sample
        gain_flat = box_gain_clean.view(B, HW)
        topM_indices = gain_flat.topk(M, dim=1)[1]  # (B, M)

        candidate_mask = torch.zeros(B, HW, dtype=torch.bool, device=device)
        candidate_mask.scatter_(1, topM_indices, True)
        candidate_mask = candidate_mask.view(B, 1, H, W)

        # -------------------------------------------------------------
        # Stage 2: Coalition Initialization (Top-K box-aware pixels)
        # -------------------------------------------------------------
        topK_indices = gain_flat.topk(effective_k, dim=1)[1]  # (B, K)
        support_mask_flat = torch.zeros(B, HW, dtype=torch.bool, device=device)
        support_mask_flat.scatter_(1, topK_indices, True)
        support_mask = support_mask_flat.view(B, 1, H, W)

        # -------------------------------------------------------------
        # Stage 3: Initial Fixed-Support RGB Optimization
        # -------------------------------------------------------------
        init_delta = torch.zeros_like(orig_x)
        delta_S, margin_S, fwd, bwd = self._optimize_fixed_support(
            orig_x, y, support_mask, init_delta, num_steps=self.inner_steps
        )
        forward_evals += fwd
        backward_evals += bwd

        with torch.no_grad():
            x_adv_curr = torch.clamp(orig_x + delta_S, 0.0, 1.0)
            logits_curr = self.model(x_adv_curr)
            forward_evals += 1
            best_succ = (logits_curr.argmax(dim=1) != y)
            best_margin = self._compute_margin(logits_curr, y)
            best_delta = delta_S.clone()
            best_support_mask = support_mask.clone()

        # -------------------------------------------------------------
        # Stages 4 - 7: Coalition Refinement (Support Exchanges)
        # -------------------------------------------------------------
        for step in range(self.steps):
            # Check if all samples are already successful
            if best_succ.all() and not self.drop_and_repair:
                break

            # Gradient at current coalition state x_S = x + delta_S
            x_curr = (orig_x + delta_S).detach().requires_grad_(True)
            logits_S = self.model(x_curr)
            forward_evals += 1
            loss_S = self._compute_loss(logits_S, y).sum()
            self.model.zero_grad()
            loss_S.backward()
            backward_evals += 1

            grad_S = x_curr.grad if x_curr.grad is not None else torch.zeros_like(x_curr)
            box_gain_S = self._compute_box_aware_gain(x_curr, grad_S)  # (B, 1, H, W)

            # --- Stage 4: Conditional Coalition Gain for inactive candidates ---
            # Score candidate pixels in C \ S
            gain_S_flat = box_gain_S.view(B, HW)
            in_candidate_out_support = candidate_mask.view(B, HW) & (~support_mask.view(B, HW))

            cand_scores = gain_S_flat.clone()
            cand_scores[~in_candidate_out_support] = float("-inf")
            j_star = cand_scores.argmax(dim=1)  # (B,) best candidate to add

            # --- Stage 5: Active Pixel Redundancy for pixels in S ---
            # Score active removal cost: R_i = (grad_S * delta_S).sum(dim=C)
            redundancy = (grad_S * delta_S).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            red_flat = redundancy.view(B, HW)
            red_scores = red_flat.clone()
            red_scores[~support_mask.view(B, HW)] = float("inf")
            i_star = red_scores.argmin(dim=1)  # (B,) weakest pixel to remove

            # --- Stage 6: Support Exchange Proposal (1-out / 1-in) ---
            new_support_flat = support_mask.view(B, HW).clone()
            b_idx = torch.arange(B, device=device)
            new_support_flat[b_idx, i_star] = False
            new_support_flat[b_idx, j_star] = True
            new_support_mask = new_support_flat.view(B, 1, H, W)

            # Warm-start RGB values for new candidate j_star based on grad_S
            warm_delta = delta_S.clone()
            warm_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0

            grad_j = grad_S.reshape(B, C, HW)[b_idx, :, j_star]
            x_j = orig_x.reshape(B, C, HW)[b_idx, :, j_star]
            target_val = torch.where(grad_j > 0, 1.0 - x_j, -x_j)
            warm_delta.reshape(B, C, HW)[b_idx, :, j_star] = self.alpha * target_val.sign()

            # Re-optimize RGB on proposed support
            proposed_delta, proposed_margin, fwd, bwd = self._optimize_fixed_support(
                orig_x, y, new_support_mask, warm_delta, num_steps=self.repair_steps
            )
            forward_evals += fwd
            backward_evals += bwd

            # --- Stage 7: Accept / Reject Criterion ---
            accept_mask = proposed_margin > (margin_S + 1e-4)

            if accept_mask.any():
                acc_b = accept_mask.view(B, 1, 1, 1)
                support_mask = torch.where(acc_b, new_support_mask, support_mask)
                delta_S = torch.where(acc_b, proposed_delta, delta_S)
                margin_S = torch.where(accept_mask, proposed_margin, margin_S)

                # Track best global solution
                with torch.no_grad():
                    x_adv_prop = torch.clamp(orig_x + delta_S, 0.0, 1.0)
                    logits_prop = self.model(x_adv_prop)
                    forward_evals += 1
                    succ_prop = (logits_prop.argmax(dim=1) != y)

                    improved = (succ_prop & ~best_succ) | ((succ_prop == best_succ) & (margin_S > best_margin))
                    if improved.any():
                        imp_b = improved.view(B, 1, 1, 1)
                        best_succ = best_succ | succ_prop
                        best_margin = torch.where(improved, margin_S, best_margin)
                        best_delta = torch.where(imp_b, delta_S, best_delta)
                        best_support_mask = torch.where(imp_b, support_mask, best_support_mask)

            # Optional Pair Exploration (2-out / 2-in)
            if self.pair_exploration and (step + 1) % self.pair_search_every == 0:
                # Top-2 candidates vs Bottom-2 active pixels
                cand_scores_2 = cand_scores.clone()
                cand_scores_2[b_idx, j_star] = float("-inf")
                j_star2 = cand_scores_2.argmax(dim=1)

                red_scores_2 = red_scores.clone()
                red_scores_2[b_idx, i_star] = float("inf")
                i_star2 = red_scores_2.argmin(dim=1)

                pair_support_flat = support_mask.view(B, HW).clone()
                pair_support_flat[b_idx, i_star] = False
                pair_support_flat[b_idx, i_star2] = False
                pair_support_flat[b_idx, j_star] = True
                pair_support_flat[b_idx, j_star2] = True
                pair_support_mask = pair_support_flat.view(B, 1, H, W)

                pair_delta, pair_margin, fwd, bwd = self._optimize_fixed_support(
                    orig_x, y, pair_support_mask, delta_S, num_steps=self.repair_steps
                )
                forward_evals += fwd
                backward_evals += bwd

                pair_accept = pair_margin > (margin_S + 1e-4)
                if pair_accept.any():
                    p_acc_b = pair_accept.view(B, 1, 1, 1)
                    support_mask = torch.where(p_acc_b, pair_support_mask, support_mask)
                    delta_S = torch.where(p_acc_b, pair_delta, delta_S)
                    margin_S = torch.where(pair_accept, pair_margin, margin_S)

        # -------------------------------------------------------------
        # Stage 8: Drop-and-Repair Support Minimization (for successful samples)
        # -------------------------------------------------------------
        final_delta = best_delta.clone()
        # -------------------------------------------------------------
        # Stage 8: Batched Drop-and-Repair Support Minimization
        # -------------------------------------------------------------
        final_delta = best_delta.clone()
        final_support = best_support_mask.clone()

        if self.drop_and_repair:
            with torch.no_grad():
                x_adv_final = torch.clamp(orig_x + final_delta, 0.0, 1.0)
                logits_final = self.model(x_adv_final)
                forward_evals += 1
                succ = (logits_final.argmax(dim=1) != y)

            active_counts = final_support.view(B, HW).sum(dim=1)
            b_idx = torch.arange(B, device=device)

            # Batched drop-and-repair iterations (up to K passes)
            for pass_idx in range(self.k):
                active_mask = (active_counts > 1) & succ
                if not active_mask.any():
                    break

                # Compute removal cost redundancy across the batch in a single pass
                with torch.enable_grad():
                    x_curr = (orig_x + final_delta).detach().requires_grad_(True)
                    loss_single = self._compute_loss(self.model(x_curr), y).sum()
                    forward_evals += 1
                    backward_evals += 1
                    loss_single.backward()
                    g_curr = x_curr.grad if x_curr.grad is not None else torch.zeros_like(x_curr)

                redundancy = (g_curr * final_delta).sum(dim=1, keepdim=True)  # (B, 1, H, W)
                red_flat = redundancy.view(B, HW).clone()
                red_flat[~final_support.view(B, HW)] = float("inf")
                i_star = red_flat.argmin(dim=1)  # (B,) weakest active pixel per sample

                # Proposed support dropping i_star for active samples
                test_supp_flat = final_support.view(B, HW).clone()
                test_supp_flat[b_idx, i_star] = False
                test_support = test_supp_flat.view(B, 1, H, W)

                test_delta = final_delta.clone()
                test_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0

                # Check direct success in batch
                with torch.no_grad():
                    x_test = torch.clamp(orig_x + test_delta, 0.0, 1.0)
                    l_test = self.model(x_test)
                    forward_evals += 1
                    direct_succ = (l_test.argmax(dim=1) != y) & active_mask

                # For samples where direct success failed, run repaired optimization in batch
                repair_mask = active_mask & (~direct_succ)
                repaired_delta = test_delta.clone()
                repair_succ = torch.zeros(B, dtype=torch.bool, device=device)

                if repair_mask.any():
                    rep_delta, _, fwd, bwd = self._optimize_fixed_support(
                        orig_x, y, test_support, test_delta, num_steps=self.repair_steps
                    )
                    forward_evals += fwd
                    backward_evals += bwd
                    with torch.no_grad():
                        x_rep = torch.clamp(orig_x + rep_delta, 0.0, 1.0)
                        l_rep = self.model(x_rep)
                        forward_evals += 1
                        repair_succ = (l_rep.argmax(dim=1) != y) & repair_mask
                    repaired_delta = rep_delta

                drop_succ = direct_succ | repair_succ
                if drop_succ.any():
                    d_mask = drop_succ.view(B, 1, 1, 1)
                    final_support = torch.where(d_mask, test_support, final_support)
                    chosen_delta = torch.where(direct_succ.view(B, 1, 1, 1), test_delta, repaired_delta)
                    final_delta = torch.where(d_mask, chosen_delta, final_delta)
                    active_counts = final_support.view(B, HW).sum(dim=1)
                else:
                    # No active sample could be pruned further in this pass
                    break

        # Enforce exact L0 budget and box clipping as safety invariant
        x_adv_out = torch.clamp(orig_x + final_delta * final_support, 0.0, 1.0)

        # Strict safety check: if any sample accidentally exceeded k, apply project_l0
        l0_check = compute_spatial_l0(x_adv_out - orig_x)
        if (l0_check > self.k).any():
            over_mask = (l0_check > self.k).view(B, 1, 1, 1)
            bounded_adv = project_l0(x_adv_out - orig_x, self.k) + orig_x
            x_adv_out = torch.where(over_mask, bounded_adv, x_adv_out)

        return AttackOutput(
            x_adv=x_adv_out,
            forward_evals=forward_evals,
            backward_evals=backward_evals,
            queries=forward_evals,
            metadata={
                "k": self.k,
                "steps": self.steps,
                "loss_fn": self.loss_fn,
                "candidate_pool_size": M,
            }
        )
