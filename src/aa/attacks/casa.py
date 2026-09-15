import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from aa.attacks.base import Attack, AttackOutput
from aa.metrics import compute_spatial_l0, exact_spatial_topk_mask, project_l0


class CoalitionSparseAttack(Attack):
    """
    CASA — Coalition-Aware Sparse Adversarial Attack (Public Upgrade Edition).

    Key Innovations:
    1. Sparse support is evaluated as a coalition: candidate pixel utility depends on current support,
       measured via conditional gain: Delta(j | S) = F(S U {j}) - F(S).
    2. Candidate screening via Box-Aware Potential Gain A_i at clean image with dynamic pool size (M >= 4K).
    3. Box-Extremal Initialization & Adaptive Inner Optimization: Initial perturbation is placed directly
       at extremal box boundaries, followed by decaying multi-step gradient ascent.
    4. Anti-Cycling Tabu Search: Swaps rejected by the exact margin criterion are tabu-masked to prevent
       search stagnation and encourage exploration of diverse pixel coalitions.
    5. Scale-Invariant DLR Loss: Maximizes Difference-of-Logits Ratio to prevent gradient vanishing
       from logit explosion.
    6. Gradient-Guided Corner Traversal (GCT) for extreme sparsity (K <= 2): Evaluates discrete RGB
       cube vertices on top-ranked gradient pixels to eliminate local minima.
    7. Spatial Non-Maximum Suppression (Spatial NMS) for coalition initialization (K >= 2): Disperses
       initial coalition across distinct receptive fields, avoiding redundant adjacent pixels.
    8. Dynamic Candidate Pool Refresh & Final Support Polish: Reactivates salient pixels and settles margins.
    9. Drop-and-Repair support minimization: Prunes successful adversarial images to minimal L0.
    """

    CORNERS = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)

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

    def _apply_spatial_nms(
        self,
        gain_flat: torch.Tensor,
        H: int,
        W: int,
        k: int,
        radius: int = 1,
    ) -> torch.Tensor:
        """
        Selects top-k pixel indices using spatial non-maximum suppression
        to disperse initial coalition members across diverse receptive fields.
        """
        B, HW = gain_flat.shape
        gain_2d = gain_flat.view(B, H, W).clone()
        selected = []
        for _ in range(k):
            flat = gain_2d.view(B, HW)
            best_idx = flat.argmax(dim=1)
            selected.append(best_idx)
            bh = best_idx // W
            bw = best_idx % W
            for b in range(B):
                h_min = max(0, bh[b].item() - radius)
                h_max = min(H, bh[b].item() + radius + 1)
                w_min = max(0, bw[b].item() - radius)
                w_max = min(W, bw[b].item() + radius + 1)
                gain_2d[b, h_min:h_max, w_min:w_max] = 0.0
        return torch.stack(selected, dim=1)

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

    def _compute_dlr_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Difference of Logits Ratio (DLR) loss (Croce & Hein 2020):
        DLR(x, y) = (z_other - z_y) / (z_{pi_1} - z_{pi_3} + eps)
        Invariant to shift and scale of logits, preventing gradient saturation.
        """
        B, C = logits.shape
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        z_p1 = sorted_logits[:, 0]
        z_p3 = sorted_logits[:, 2]
        denom = z_p1 - z_p3 + 1e-6

        one_hot = F.one_hot(y, num_classes=C).bool()
        z_y = logits[one_hot]

        logits_other = logits.clone()
        logits_other[one_hot] = float("-inf")
        z_other = logits_other.max(dim=1)[0]

        return (z_other - z_y) / denom

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes loss to maximize."""
        if self.loss_fn == "dlr":
            return self._compute_dlr_loss(logits, y)
        elif self.loss_fn == "margin":
            return self._compute_margin(logits, y)
        elif self.loss_fn == "ce":
            return F.cross_entropy(logits, y, reduction="none")
        else:
            raise ValueError(f"Unknown loss_fn '{self.loss_fn}'. Options: ['dlr', 'margin', 'ce']")

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
        Uses adaptive decaying step size to traverse box [0, 1] and settle accurately.
        Returns: (best_delta, best_margin, fwd_evals, bwd_evals)
        """
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
                # Decaying step size for rapid coverage and fine refinement
                cur_alpha = self.alpha * (0.85 ** step)
                step_delta = curr_delta + cur_alpha * grad.sign() * support_mask
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

        # Candidate pool size M: dynamically sized and guaranteed strictly > k
        effective_k = min(self.k, HW)
        if self.candidate_pool_size is not None:
            M = max(self.candidate_pool_size, effective_k + 16)
        else:
            M = max(64, effective_k * self.candidate_pool_multiplier)
        M = min(M, HW)

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
        b_idx = torch.arange(B, device=device)

        best_delta = torch.zeros_like(orig_x)
        with torch.no_grad():
            best_succ = (logits_clean.argmax(dim=1) != y)
            best_margin = self._compute_margin(logits_clean, y)
            best_support_mask = torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)

        # -------------------------------------------------------------
        # Stage 1.5: Gradient-Guided Corner Traversal (GCT) for K <= 2
        # -------------------------------------------------------------
        if self.enable_gct and effective_k in (1, 2):
            corners = self.CORNERS.to(device)
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
                        m_test = self._compute_margin(l_test, y)
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
        topM_indices = gain_flat.topk(M, dim=1)[1]  # (B, M)
        candidate_mask = torch.zeros(B, HW, dtype=torch.bool, device=device)
        candidate_mask.scatter_(1, topM_indices, True)
        candidate_mask = candidate_mask.view(B, 1, H, W)

        if self.spatial_nms and effective_k >= 2:
            topK_indices = self._apply_spatial_nms(gain_flat, H, W, effective_k, radius=self.nms_radius)
        else:
            topK_indices = gain_flat.topk(effective_k, dim=1)[1]  # (B, K)

        support_mask_flat = torch.zeros(B, HW, dtype=torch.bool, device=device)
        support_mask_flat.scatter_(1, topK_indices, True)
        support_mask = support_mask_flat.view(B, 1, H, W)

        # -------------------------------------------------------------
        # Stage 3: Initial Fixed-Support RGB Optimization (Box-Extremal Warm-Start)
        # -------------------------------------------------------------
        # Push pixels in support directly to extremal boundaries [0, 1] along gradient
        init_delta = torch.where(grad_clean > 0, 1.0 - orig_x, -orig_x) * support_mask
        delta_S, margin_S, fwd, bwd = self._optimize_fixed_support(
            orig_x, y, support_mask, init_delta, num_steps=self.inner_steps
        )
        forward_evals += fwd
        backward_evals += bwd

        with torch.no_grad():
            x_adv_curr = torch.clamp(orig_x + delta_S, 0.0, 1.0)
            logits_curr = self.model(x_adv_curr)
            forward_evals += 1
            s_curr = (logits_curr.argmax(dim=1) != y)
            m_curr = self._compute_margin(logits_curr, y)
            imp = (s_curr & (~best_succ)) | ((s_curr == best_succ) & (m_curr > best_margin))
            if imp.any():
                imp_b = imp.view(B, 1, 1, 1)
                best_succ = best_succ | s_curr
                best_margin = torch.where(imp, m_curr, best_margin)
                best_delta = torch.where(imp_b, delta_S, best_delta)
                best_support_mask = torch.where(imp_b, support_mask, best_support_mask)

        # Tabu mask per sample to prevent cycling over rejected candidates
        tabu_mask = torch.zeros(B, HW, dtype=torch.bool, device=device)

        # -------------------------------------------------------------
        # Stages 4 - 7: Coalition Refinement (Support Exchanges)
        # -------------------------------------------------------------
        for step in range(self.steps):
            # Check if all samples are already successful
            if best_succ.all():
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

            # Dynamic candidate pool refresh: incorporate newly prominent pixels into candidate mask
            if (step + 1) % 4 == 0:
                new_top_cands = box_gain_S.view(B, HW).topk(M // 2, dim=1)[1]
                cand_flat = candidate_mask.view(B, HW)
                cand_flat.scatter_(1, new_top_cands, True)
                candidate_mask = cand_flat.view(B, 1, H, W)

            # --- Stage 4: Conditional Coalition Gain for inactive candidates ---
            gain_S_flat = box_gain_S.view(B, HW)
            in_candidate_out_support = candidate_mask.view(B, HW) & (~support_mask.view(B, HW)) & (~tabu_mask)

            cand_scores = gain_S_flat.clone()
            cand_scores[~in_candidate_out_support] = float("-inf")

            # If all candidates are tabu for a sample, reset tabu mask to continue exploring
            no_cand = (cand_scores.max(dim=1)[0] == float("-inf"))
            if no_cand.any():
                tabu_mask[no_cand] = False
                in_candidate_out_support = candidate_mask.view(B, HW) & (~support_mask.view(B, HW))
                cand_scores = gain_S_flat.clone()
                cand_scores[~in_candidate_out_support] = float("-inf")

            j_star = cand_scores.argmax(dim=1)  # (B,) best candidate to add

            # --- Stage 5: Active Pixel Redundancy for pixels in S ---
            redundancy = (grad_S * delta_S).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            red_flat = redundancy.view(B, HW)
            red_scores = red_flat.clone()
            red_scores[~support_mask.view(B, HW)] = float("inf")
            i_star = red_scores.argmin(dim=1)  # (B,) weakest pixel to remove

            # --- Stage 6: Support Exchange Proposal (1-out / 1-in) ---
            new_support_flat = support_mask.view(B, HW).clone()
            new_support_flat[b_idx, i_star] = False
            new_support_flat[b_idx, j_star] = True
            new_support_mask = new_support_flat.view(B, 1, H, W)

            # Warm-start RGB values for new candidate j_star based on full boundary reach
            warm_delta = delta_S.clone()
            warm_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0

            grad_j = grad_S.reshape(B, C, HW)[b_idx, :, j_star]
            x_j = orig_x.reshape(B, C, HW)[b_idx, :, j_star]
            target_val = torch.where(grad_j > 0, 1.0 - x_j, -x_j)
            warm_delta.reshape(B, C, HW)[b_idx, :, j_star] = target_val

            # Re-optimize RGB on proposed support
            proposed_delta, proposed_margin, fwd, bwd = self._optimize_fixed_support(
                orig_x, y, new_support_mask, warm_delta, num_steps=self.repair_steps
            )
            forward_evals += fwd
            backward_evals += bwd

            # --- Stage 7: Accept / Reject Criterion with Adaptive Batch Swap ---
            accept_mask = proposed_margin > (margin_S + 1e-4)
            rejected_mask = ~accept_mask

            # If 1-swap is rejected and K >= 4: attempt joint 2-pixel swap fallback to break synergy trap!
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

                pair_delta, pair_margin, fwd, bwd = self._optimize_fixed_support(
                    orig_x, y, pair_support_mask, pair_warm_delta, num_steps=self.repair_steps
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

            # Record rejected candidates into tabu mask to prevent infinite cycling
            if rejected_mask.any():
                tabu_mask[rejected_mask, j_star[rejected_mask]] = True

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
            if self.pair_exploration and (step + 1) % self.pair_search_every == 0 and effective_k >= 2:
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

                pair_delta, pair_margin, fwd, bwd = self._optimize_fixed_support(
                    orig_x, y, pair_support_mask, pair_warm_delta, num_steps=self.repair_steps
                )
                forward_evals += fwd
                backward_evals += bwd

                pair_accept = pair_margin > (margin_S + 1e-4)
                if pair_accept.any():
                    p_acc_b = pair_accept.view(B, 1, 1, 1)
                    support_mask = torch.where(p_acc_b, pair_support_mask, support_mask)
                    delta_S = torch.where(p_acc_b, pair_delta, delta_S)
                    margin_S = torch.where(pair_accept, pair_margin, margin_S)

                    with torch.no_grad():
                        x_adv_pair = torch.clamp(orig_x + pair_delta, 0.0, 1.0)
                        succ_pair = (self.model(x_adv_pair).argmax(dim=1) != y)
                        best_succ = best_succ | succ_pair

                    best_delta = torch.where(p_acc_b, pair_delta, best_delta)
                    best_support_mask = torch.where(p_acc_b, pair_support_mask, best_support_mask)
                    best_margin = torch.where(pair_accept, pair_margin, best_margin)

        # Final Polish on winning support for un-fooled samples
        not_yet_succ = ~best_succ
        if not_yet_succ.any():
            polished_delta, polished_margin, fwd, bwd = self._optimize_fixed_support(
                orig_x, y, best_support_mask, best_delta, num_steps=self.inner_steps
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
            with torch.no_grad():
                x_adv_final = torch.clamp(orig_x + final_delta, 0.0, 1.0)
                logits_final = self.model(x_adv_final)
                forward_evals += 1
                succ = (logits_final.argmax(dim=1) != y)

            active_counts = final_support.view(B, HW).sum(dim=1)

            # Batched drop-and-repair iterations (up to min(k, 6) passes)
            max_drop_passes = min(self.k, 6)
            for pass_idx in range(max_drop_passes):
                active_mask = (active_counts > 1) & succ
                if not active_mask.any():
                    break

                with torch.enable_grad():
                    x_curr = (orig_x + final_delta).detach().requires_grad_(True)
                    loss_single = self._compute_loss(self.model(x_curr), y).sum()
                    forward_evals += 1
                    backward_evals += 1
                    loss_single.backward()
                    g_curr = x_curr.grad if x_curr.grad is not None else torch.zeros_like(x_curr)

                redundancy = (g_curr * final_delta).sum(dim=1, keepdim=True)
                red_flat = redundancy.reshape(B, HW).clone()
                red_flat[~final_support.reshape(B, HW)] = float("inf")

                sorted_red_indices = red_flat.argsort(dim=1)  # (B, HW)

                any_pruned_in_pass = False
                max_rank_to_try = min(3, self.k)

                for rank_k in range(max_rank_to_try):
                    i_star = sorted_red_indices[:, rank_k]

                    test_supp_flat = final_support.reshape(B, HW).clone()
                    test_supp_flat[b_idx, i_star] = False
                    test_support = test_supp_flat.reshape(B, 1, H, W)

                    test_delta = final_delta.clone()
                    test_delta.reshape(B, C, HW)[b_idx, :, i_star] = 0.0

                    with torch.no_grad():
                        x_test = torch.clamp(orig_x + test_delta, 0.0, 1.0)
                        l_test = self.model(x_test)
                        forward_evals += 1
                        direct_succ = (l_test.argmax(dim=1) != y) & active_mask

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
                        d_mask = drop_succ.reshape(B, 1, 1, 1)
                        final_support = torch.where(d_mask, test_support, final_support)
                        chosen_delta = torch.where(direct_succ.reshape(B, 1, 1, 1), test_delta, repaired_delta)
                        final_delta = torch.where(d_mask, chosen_delta, final_delta)
                        active_counts = final_support.reshape(B, HW).sum(dim=1)
                        any_pruned_in_pass = True
                        break

                if not any_pruned_in_pass:
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
