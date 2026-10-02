import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from aa.attacks.base import Attack, AttackOutput
from aa.metrics import compute_spatial_l0, project_l0
from aa.attacks.casa.objective import CoalitionObjective
from aa.attacks.casa.candidate import CandidateGenerator, compute_box_aware_gain
from aa.attacks.casa.initialization import apply_spatial_nms, CORNERS
from aa.attacks.casa.value_optimization import optimize_fixed_support
from aa.attacks.casa.compression import drop_and_repair_support


class CoalitionSparseAttack(Attack):
    """
    CASA — Coalition-Aware Sparse Adversarial Attack (Modularized Architecture).
    """

    def __init__(
        self,
        model: nn.Module,
        k: int = 16,
        steps: int = 20,
        inner_steps: int = 10,
        repair_steps: int = 4,
        alpha: float = 0.25,
        candidate_pool_size: Optional[int] = None,
        candidate_pool_multiplier: int = 4,
        loss_fn: str = "dlr",
        drop_and_repair: bool = True,
        pair_exploration: bool = True,
        pair_search_every: int = 3,
        enable_gct: bool = True,
        gct_pixels: Optional[int] = None,
        spatial_nms: bool = True,
        nms_radius: int = 1,
        adaptive_batch_swap: bool = True,
    ):
        self.model = model
        self.k = k
        self.steps = steps
        self.inner_steps = inner_steps
        self.repair_steps = repair_steps
        self.alpha = alpha
        self.candidate_pool_size = candidate_pool_size
        self.candidate_pool_multiplier = candidate_pool_multiplier
        self.loss_fn = loss_fn.lower()
        self.drop_and_repair = drop_and_repair
        self.pair_exploration = pair_exploration
        self.pair_search_every = pair_search_every
        self.enable_gct = enable_gct
        self.gct_pixels = gct_pixels
        self.spatial_nms = spatial_nms
        self.nms_radius = nms_radius
        self.adaptive_batch_swap = adaptive_batch_swap

        self.objective = CoalitionObjective(loss_fn=self.loss_fn)
        self.candidate_gen = CandidateGenerator(multiplier=self.candidate_pool_multiplier)

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        B, C, H, W = x.shape
        HW = H * W
        orig_x = x.clone().detach()
        y = y.clone().detach()

        forward_evals = 0
        backward_evals = 0

        effective_k = min(self.k, HW)
        M = self.candidate_gen.get_pool_size(effective_k, HW, fixed_size=self.candidate_pool_size)

        # -------------------------------------------------------------
        # Stage 1: Candidate Generation (Box-Aware Potential Gain at clean x)
        # -------------------------------------------------------------
        x_clean = orig_x.clone().requires_grad_(True)
        logits_clean = self.model(x_clean)
        forward_evals += 1
        loss_clean = self.objective.compute_loss(logits_clean, y).sum()
        self.model.zero_grad()
        loss_clean.backward()
        backward_evals += 1

        grad_clean = x_clean.grad if x_clean.grad is not None else torch.zeros_like(x_clean)
        box_gain_clean = compute_box_aware_gain(orig_x, grad_clean)

        gain_flat = box_gain_clean.view(B, HW)
        b_idx = torch.arange(B, device=device)

        best_delta = torch.zeros_like(orig_x)
        with torch.no_grad():
            best_succ = (logits_clean.argmax(dim=1) != y)
            best_margin = self.objective.compute_margin(logits_clean, y)
            best_support_mask = torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)

        # -------------------------------------------------------------
        # Stage 1.5: Gradient-Guided Corner Traversal (GCT) for K <= 2
        # -------------------------------------------------------------
        if self.enable_gct and effective_k in (1, 2):
            corners = CORNERS.to(device)
            top_p = self.gct_pixels if self.gct_pixels is not None else (12 if effective_k == 1 else 6)
            top_pixels = gain_flat.topk(min(top_p, HW), dim=1)[1]
            for p_idx in range(top_pixels.shape[1]):
                if best_succ.all():
                    break
                pix = top_pixels[:, p_idx]
                for c in corners:
                    x_test = orig_x.clone()
                    x_test.view(B, C, HW)[b_idx, :, pix] = c.view(1, 3)
                    with torch.no_grad():
                        l_test = self.model(x_test)
                        forward_evals += 1
                        m_test = self.objective.compute_margin(l_test, y)
                        s_test = (l_test.argmax(dim=1) != y)
                        improved = (s_test & (~best_succ)) | ((s_test == best_succ) & (m_test > best_margin))
                        if improved.any():
                            imp_b = improved.view(B, 1, 1, 1)
                            best_succ = best_succ | s_test
                            best_margin = torch.where(improved, m_test, best_margin)
                            best_delta = torch.where(imp_b, x_test - orig_x, best_delta)
                            supp_pix = torch.zeros(B, HW, dtype=torch.bool, device=device)
                            supp_pix[b_idx, pix] = True
                            best_support_mask = torch.where(imp_b, supp_pix.view(B, 1, H, W), best_support_mask)

        # -------------------------------------------------------------
        # Stage 2: Candidate Pool & Coalition Initialization
        # -------------------------------------------------------------
        topM_indices = gain_flat.topk(M, dim=1)[1]
        candidate_mask = torch.zeros(B, HW, dtype=torch.bool, device=device)
        candidate_mask.scatter_(1, topM_indices, True)
        candidate_mask = candidate_mask.view(B, 1, H, W)

        if self.spatial_nms and effective_k >= 2:
            topK_indices = apply_spatial_nms(gain_flat, H, W, effective_k, radius=self.nms_radius)
        else:
            topK_indices = gain_flat.topk(effective_k, dim=1)[1]

        support_mask_flat = torch.zeros(B, HW, dtype=torch.bool, device=device)
        support_mask_flat.scatter_(1, topK_indices, True)
        support_mask = support_mask_flat.view(B, 1, H, W)

        # -------------------------------------------------------------
        # Stage 3: Initial Fixed-Support RGB Optimization (Box-Extremal Warm-Start)
        # -------------------------------------------------------------
        init_delta = torch.where(grad_clean > 0, 1.0 - orig_x, -orig_x) * support_mask
        delta_S, margin_S, fwd, bwd = optimize_fixed_support(
            self.model, self.objective, orig_x, y, support_mask, init_delta,
            num_steps=self.inner_steps, alpha=self.alpha
        )
        forward_evals += fwd
        backward_evals += bwd

        with torch.no_grad():
            x_adv_curr = torch.clamp(orig_x + delta_S, 0.0, 1.0)
            logits_curr = self.model(x_adv_curr)
            forward_evals += 1
            s_curr = (logits_curr.argmax(dim=1) != y)
            m_curr = self.objective.compute_margin(logits_curr, y)
            imp = (s_curr & (~best_succ)) | ((s_curr == best_succ) & (m_curr > best_margin))
            if imp.any():
                imp_b = imp.view(B, 1, 1, 1)
                best_succ = best_succ | s_curr
                best_margin = torch.where(imp, m_curr, best_margin)
                best_delta = torch.where(imp_b, delta_S, best_delta)
                best_support_mask = torch.where(imp_b, support_mask, best_support_mask)

        tabu_mask = torch.zeros(B, HW, dtype=torch.bool, device=device)

        # -------------------------------------------------------------
        # Stages 4 - 7: Coalition Refinement (Support Exchanges)
        # -------------------------------------------------------------
        for step in range(self.steps):
            if best_succ.all():
                break

            x_curr = (orig_x + delta_S).detach().requires_grad_(True)
            logits_S = self.model(x_curr)
            forward_evals += 1
            loss_S = self.objective.compute_loss(logits_S, y).sum()
            self.model.zero_grad()
            loss_S.backward()
            backward_evals += 1

            grad_S = x_curr.grad if x_curr.grad is not None else torch.zeros_like(x_curr)
            box_gain_S = compute_box_aware_gain(x_curr, grad_S)

            if (step + 1) % 4 == 0:
                new_top_cands = box_gain_S.view(B, HW).topk(M // 2, dim=1)[1]
                cand_flat = candidate_mask.view(B, HW)
                cand_flat.scatter_(1, new_top_cands, True)
                candidate_mask = cand_flat.view(B, 1, H, W)

            gain_S_flat = box_gain_S.view(B, HW)
            in_candidate_out_support = candidate_mask.view(B, HW) & (~support_mask.view(B, HW)) & (~tabu_mask)

            cand_scores = gain_S_flat.clone()
            cand_scores[~in_candidate_out_support] = float("-inf")

            no_cand = (cand_scores.max(dim=1)[0] == float("-inf"))
            if no_cand.any():
                tabu_mask[no_cand] = False
                in_candidate_out_support = candidate_mask.view(B, HW) & (~support_mask.view(B, HW))
                cand_scores = gain_S_flat.clone()
                cand_scores[~in_candidate_out_support] = float("-inf")

            j_star = cand_scores.argmax(dim=1)

            redundancy = (grad_S * delta_S).sum(dim=1, keepdim=True)
            red_flat = redundancy.view(B, HW)
            red_scores = red_flat.clone()
            red_scores[~support_mask.view(B, HW)] = float("inf")
            i_star = red_scores.argmin(dim=1)

            new_support_flat = support_mask.view(B, HW).clone()
            new_support_flat[b_idx, i_star] = False
            new_support_flat[b_idx, j_star] = True
            new_support_mask = new_support_flat.view(B, 1, H, W)

            warm_delta = delta_S.clone()
            warm_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0

            grad_j = grad_S.reshape(B, C, HW)[b_idx, :, j_star]
            x_j = orig_x.reshape(B, C, HW)[b_idx, :, j_star]
            target_val = torch.where(grad_j > 0, 1.0 - x_j, -x_j)
            warm_delta.reshape(B, C, HW)[b_idx, :, j_star] = target_val

            proposed_delta, proposed_margin, fwd, bwd = optimize_fixed_support(
                self.model, self.objective, orig_x, y, new_support_mask, warm_delta,
                num_steps=self.repair_steps, alpha=self.alpha
            )
            forward_evals += fwd
            backward_evals += bwd

            accept_mask = proposed_margin > (margin_S + 1e-4)
            rejected_mask = ~accept_mask

            if self.adaptive_batch_swap and rejected_mask.any() and effective_k >= 4:
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

                pair_warm_delta = delta_S.clone()
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, i_star2] = 0.0
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, j_star] = target_val
                grad_j2 = grad_S.reshape(B, C, HW)[b_idx, :, j_star2]
                x_j2 = orig_x.reshape(B, C, HW)[b_idx, :, j_star2]
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, j_star2] = torch.where(grad_j2 > 0, 1.0 - x_j2, -x_j2)

                pair_delta, pair_margin, fwd, bwd = optimize_fixed_support(
                    self.model, self.objective, orig_x, y, pair_support_mask, pair_warm_delta,
                    num_steps=self.repair_steps, alpha=self.alpha
                )
                forward_evals += fwd
                backward_evals += bwd

                pair_acc = (pair_margin > (margin_S + 1e-4)) & rejected_mask
                if pair_acc.any():
                    accept_mask = accept_mask | pair_acc
                    rejected_mask = rejected_mask & (~pair_acc)
                    p_acc_b = pair_acc.view(B, 1, 1, 1)
                    proposed_delta = torch.where(p_acc_b, pair_delta, proposed_delta)
                    proposed_margin = torch.where(pair_acc, pair_margin, proposed_margin)
                    new_support_mask = torch.where(p_acc_b, pair_support_mask, new_support_mask)

            if rejected_mask.any():
                tabu_mask[rejected_mask, j_star[rejected_mask]] = True

            if accept_mask.any():
                acc_b = accept_mask.view(B, 1, 1, 1)
                support_mask = torch.where(acc_b, new_support_mask, support_mask)
                delta_S = torch.where(acc_b, proposed_delta, delta_S)
                margin_S = torch.where(accept_mask, proposed_margin, margin_S)

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

            # Optional Pair Exploration (2-out / 2-in) to rescue samples where 1-swap was rejected
            if self.pair_exploration and (step + 1) % self.pair_search_every == 0 and effective_k >= 2 and rejected_mask.any():
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

                pair_warm_delta = delta_S.clone()
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, i_star2] = 0.0

                grad_j1 = grad_S.reshape(B, C, HW)[b_idx, :, j_star]
                x_j1 = orig_x.reshape(B, C, HW)[b_idx, :, j_star]
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, j_star] = torch.where(grad_j1 > 0, 1.0 - x_j1, -x_j1)

                grad_j2 = grad_S.reshape(B, C, HW)[b_idx, :, j_star2]
                x_j2 = orig_x.reshape(B, C, HW)[b_idx, :, j_star2]
                pair_warm_delta.reshape(B, C, HW)[b_idx, :, j_star2] = torch.where(grad_j2 > 0, 1.0 - x_j2, -x_j2)

                pair_delta, pair_margin, fwd, bwd = optimize_fixed_support(
                    self.model, self.objective, orig_x, y, pair_support_mask, pair_warm_delta,
                    num_steps=self.repair_steps, alpha=self.alpha
                )
                forward_evals += fwd
                backward_evals += bwd

                pair_accept = (pair_margin > (margin_S + 1e-4)) & rejected_mask
                if pair_accept.any():
                    p_acc_b = pair_accept.view(B, 1, 1, 1)
                    support_mask = torch.where(p_acc_b, pair_support_mask, support_mask)
                    delta_S = torch.where(p_acc_b, pair_delta, delta_S)
                    margin_S = torch.where(pair_accept, pair_margin, margin_S)

                    with torch.no_grad():
                        x_adv_pair = torch.clamp(orig_x + pair_delta, 0.0, 1.0)
                        succ_pair = (self.model(x_adv_pair).argmax(dim=1) != y) & pair_accept
                        best_succ = best_succ | succ_pair

                    best_delta = torch.where(p_acc_b, pair_delta, best_delta)
                    best_support_mask = torch.where(p_acc_b, pair_support_mask, best_support_mask)
                    best_margin = torch.where(pair_accept, pair_margin, best_margin)

        # Final Polish on winning support for un-fooled samples
        not_yet_succ = ~best_succ
        if not_yet_succ.any():
            polished_delta, polished_margin, fwd, bwd = optimize_fixed_support(
                self.model, self.objective, orig_x, y, best_support_mask, best_delta,
                num_steps=self.inner_steps, alpha=self.alpha
            )
            forward_evals += fwd
            backward_evals += bwd
            with torch.no_grad():
                succ_pol = (self.model(torch.clamp(orig_x + polished_delta, 0.0, 1.0)).argmax(dim=1) != y)
                best_succ = best_succ | succ_pol
                pol_imp = polished_margin > best_margin
                if pol_imp.any():
                    p_b = pol_imp.view(B, 1, 1, 1)
                    best_delta = torch.where(p_b, polished_delta, best_delta)
                    best_margin = torch.where(pol_imp, polished_margin, best_margin)

        # -------------------------------------------------------------
        # Stage 8: Drop-and-Repair Support Minimization (for successful samples)
        # -------------------------------------------------------------
        final_delta = best_delta.clone()
        final_support = best_support_mask.clone()

        if self.drop_and_repair:
            final_delta, final_support, fwd, bwd = drop_and_repair_support(
                self.model, self.objective, orig_x, y, final_delta, final_support,
                k=self.k, repair_steps=self.repair_steps, alpha=self.alpha
            )
            forward_evals += fwd
            backward_evals += bwd

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
            queries=0,
            sample_forward_evals=forward_evals * B,
            sample_backward_evals=backward_evals * B,
            metadata={
                "k": self.k,
                "steps": self.steps,
                "loss_fn": self.loss_fn,
                "candidate_pool_size": M,
                "batch_size": B,
                "model_forward_calls": forward_evals,
                "sample_forward_evals": forward_evals * B,
                "model_backward_calls": backward_evals,
                "sample_backward_evals": backward_evals * B,
                "flop_equivalent_evals": (forward_evals + 2 * backward_evals) * B,
            }
        )


CASAAttack = CoalitionSparseAttack
