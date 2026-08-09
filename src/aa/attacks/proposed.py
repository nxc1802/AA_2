import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from aa.attacks.base import Attack, AttackOutput
from aa.metrics import project_l0, exact_spatial_topk_mask, compute_spatial_l0


class FeatureExtractorAdapter:
    """Extracts internal intermediate features for feature guidance strictly without silent failure."""
    def __init__(self, model: nn.Module, layer_name: Optional[str] = None, required: bool = False):
        self.model = model
        self.layer_name = layer_name
        self.extracted_features = None
        self.hook_handle = None
        self.required = required
        self._attach_hook()

    def _attach_hook(self):
        target_layer = None
        # Unwrap DefendedModelAdapter or similar wrappers
        base_model = getattr(self.model, "model", self.model)
        if self.layer_name and hasattr(base_model, self.layer_name):
            target_layer = getattr(base_model, self.layer_name)
        elif hasattr(base_model, "layer4"):
            target_layer = base_model.layer4
        elif hasattr(base_model, "block3"):
            target_layer = base_model.block3
        elif hasattr(base_model, "features"):
            target_layer = base_model.features
        # Fallback: try on the original model object
        elif self.layer_name and hasattr(self.model, self.layer_name):
            target_layer = getattr(self.model, self.layer_name)
        elif hasattr(self.model, "layer4"):
            target_layer = self.model.layer4
        elif hasattr(self.model, "block3"):
            target_layer = self.model.block3
        elif hasattr(self.model, "features"):
            target_layer = self.model.features

        if target_layer is not None:
            def hook(module, input, output):
                self.extracted_features = output
            self.hook_handle = target_layer.register_forward_hook(hook)
        elif self.required:
            raise ValueError(
                f"Feature guidance requested (layer_name='{self.layer_name}') but no valid intermediate feature layer "
                f"('layer4', 'block3', or 'features') was found on model {type(self.model).__name__} "
                f"or its wrapped base model {type(base_model).__name__}."
            )

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class SparseFeatureAttack(Attack):
    """
    Proposed Method: Feature-Guided Collaborative Sparse Adversarial Attack with Support Pruning.

    Ablation & Method Flags:
    - feature_guidance: Add normalized intermediate representation disruption loss.
    - feature_loss_mode: "relative" (default) normalizes by ||phi(x)||^2 for cross-arch stability;
                         "absolute" normalizes by feature_dim (old behavior).
    - loss_fn: "ce" (default) uses cross-entropy; "margin" uses max_{j!=y} z_j - z_y
               (avoids CE saturation, often stronger for sparse attacks).
    - interaction: Enable spatial pixel interaction scoring (CPA / FCSA / HSA / smoothing).
    - interaction_mode: "cpa" (directional alignment), "fcsa" (functional synergy),
                        "hsa" (hypergraph centrality), "smoothing".
    - pruning: Perform iterative post-success support pruning to minimal L0.
    - continue_after_success: If True, do not freeze samples after first success; continue
                              optimizing to explore supports with smaller L0. Default False
                              (backward-compatible, success-first mode).
    """
    def __init__(
        self,
        model: nn.Module,
        k: int = 16,
        steps: int = 25,
        alpha: float = 4 / 255.0,
        feature_weight: float = 1.0,
        feature_guidance: bool = True,
        feature_loss_mode: str = "relative",
        loss_fn: str = "ce",
        interaction: bool = True,
        interaction_mode: str = "cpa",
        coop_weight: float = 0.5,
        pruning: bool = True,
        pruning_max_passes: int = 5,
        continue_after_success: bool = False,
        layer_name: Optional[str] = None,
    ):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.feature_guidance = feature_guidance
        self.feature_loss_mode = feature_loss_mode.lower()
        self.loss_fn = loss_fn.lower()
        self.interaction = interaction
        self.interaction_mode = interaction_mode.lower()
        self.coop_weight = coop_weight
        self.pruning = pruning
        self.pruning_max_passes = pruning_max_passes
        self.continue_after_success = continue_after_success
        self.layer_name = layer_name
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def _compute_spatial_interaction(self, grad: torch.Tensor) -> torch.Tensor:
        """Computes spatial interaction score based on selected interaction_mode."""
        B, C, H, W = grad.shape
        grad_mag = grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)

        if not self.interaction or self.interaction_mode == "none":
            return grad_mag

        if self.interaction_mode == "cpa":
            # Directional Cooperation: I(i) = ||g_i||_1 + coop_weight * sum_{j in N(i)} relu(cos(g_i, g_j)) * ||g_j||_1
            # Uses zero-padding + shifted slicing to avoid circular boundary artifacts from torch.roll.
            grad_norm = grad.norm(p=2, dim=1, keepdim=True) + 1e-8
            grad_unit = grad / grad_norm  # (B, C, H, W)
            coop_score = grad_mag.clone()
            # 8-connectivity neighbors via explicit padding
            # pad: (left, right, top, bottom)
            grad_unit_pad = F.pad(grad_unit, (1, 1, 1, 1), mode="constant", value=0.0)  # (B,C,H+2,W+2)
            grad_mag_pad  = F.pad(grad_mag,  (1, 1, 1, 1), mode="constant", value=0.0)  # (B,1,H+2,W+2)
            shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dh, dw in shifts:
                # Slice the padded tensor to get neighbor at offset (dh, dw)
                sh = 1 + dh  # shifted start row in padded tensor
                sw = 1 + dw  # shifted start col in padded tensor
                unit_shift = grad_unit_pad[:, :, sh:sh+H, sw:sw+W]
                mag_shift  = grad_mag_pad[:,  :, sh:sh+H, sw:sw+W]
                alignment = (grad_unit * unit_shift).sum(dim=1, keepdim=True)
                coop_score += self.coop_weight * F.relu(alignment) * mag_shift
            return coop_score

        elif self.interaction_mode == "fcsa":
            # Functional Coalition Synergy:
            # synergy(i) = relu(max over 3x3 patch - mean over 3x3 patch)
            # patch_max captures the strongest contributor; patch_mean is the average.
            # When max > mean, the pixel benefits from a strong coalition partner.
            grad_max = grad.abs().max(dim=1, keepdim=True)[0]
            indiv_contrib = grad_mag * grad_max
            patch_max  = F.max_pool2d(indiv_contrib, kernel_size=3, stride=1, padding=1)
            patch_mean = F.avg_pool2d(indiv_contrib, kernel_size=3, stride=1, padding=1)
            synergy = F.relu(patch_max - patch_mean)
            return indiv_contrib + 0.5 * synergy

        elif self.interaction_mode == "hsa":
            # Hypergraph Centrality: Multi-scale receptive fields
            w3 = F.conv2d(grad_mag, torch.ones(1, 1, 3, 3, device=grad.device), padding=1)
            w5 = F.conv2d(grad_mag, torch.ones(1, 1, 5, 5, device=grad.device), padding=2)
            w7 = F.conv2d(grad_mag, torch.ones(1, 1, 7, 7, device=grad.device), padding=3)
            d_v = w3 + w5 + w7 + 1e-8
            norm_g = grad_mag / d_v
            c3 = F.conv2d(norm_g, torch.ones(1, 1, 3, 3, device=grad.device), padding=1) * (w3 / 9.0)
            c5 = F.conv2d(norm_g, torch.ones(1, 1, 5, 5, device=grad.device), padding=2) * (w5 / 25.0)
            c7 = F.conv2d(norm_g, torch.ones(1, 1, 7, 7, device=grad.device), padding=3) * (w7 / 49.0)
            return grad_mag + c3 + c5 + c7

        elif self.interaction_mode == "smoothing":
            return grad_mag + 0.5 * F.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)

        return grad_mag

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        B, C, H, W = x.shape
        orig_x = x.clone().detach()
        y = y.clone().detach()

        feature_adapter = FeatureExtractorAdapter(
            self.model,
            layer_name=self.layer_name,
            required=self.feature_guidance
        )

        forward_evals = 0
        backward_evals = 0

        with torch.no_grad():
            out_init = self.model(orig_x)
            forward_evals += 1
            clean_features = (
                feature_adapter.extracted_features.clone().detach()
                if (self.feature_guidance and feature_adapter.extracted_features is not None)
                else None
            )
            best_loss = self.ce_loss(out_init, y)
            best_succ = (out_init.argmax(dim=1) != y)
            best_l0 = torch.where(
                best_succ,
                torch.zeros(B, device=device),
                torch.full((B,), float("inf"), device=device)
            )
            best_adv = orig_x.clone()

        delta = torch.zeros_like(orig_x)
        fooled_mask = best_succ.clone()

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_x = (orig_x + delta).clamp(0.0, 1.0).requires_grad_(True)
            outputs = self.model(adv_x)
            forward_evals += 1

            ce_l = self.ce_loss(outputs, y)

            # Loss function: CE (default) or Margin (max_{j!=y} z_j - z_y)
            if self.loss_fn == "margin":
                # Margin loss: stronger gradient signal, avoids CE saturation
                with torch.no_grad():
                    y_one_hot = torch.zeros_like(outputs)
                    y_one_hot.scatter_(1, y.unsqueeze(1), 1.0)
                logit_y = (outputs * y_one_hot).sum(dim=1)
                logit_max_other = (outputs - 1e9 * y_one_hot).max(dim=1).values
                attack_loss = logit_max_other - logit_y
            else:
                attack_loss = ce_l

            if self.feature_guidance and feature_adapter.extracted_features is not None and clean_features is not None:
                feat_diff = feature_adapter.extracted_features - clean_features
                if self.feature_loss_mode == "relative":
                    # Relative normalization: scale-invariant across layers/architectures
                    clean_norm_sq = clean_features.pow(2).flatten(1).sum(dim=1) + 1e-8
                    feat_l = feat_diff.pow(2).flatten(1).sum(dim=1) / clean_norm_sq
                else:
                    # Absolute normalization: normalize by feature dimension (legacy)
                    feat_dim = clean_features.flatten(1).size(1)
                    feat_l = feat_diff.pow(2).flatten(1).sum(dim=1) / max(1.0, float(feat_dim))
                total_loss = attack_loss + self.feature_weight * feat_l
            else:
                total_loss = attack_loss

            loss = total_loss.sum()
            self.model.zero_grad()
            loss.backward()
            backward_evals += 1

            grad = adv_x.grad.data
            score = self._compute_spatial_interaction(grad)

            score_masked = score.clone()
            # In continue_after_success mode we never mask out already-fooled samples,
            # allowing the optimization to keep exploring smaller-support solutions.
            if not self.continue_after_success:
                score_masked[fooled_mask] = -float("inf")

            support_mask = exact_spatial_topk_mask(score_masked, self.k).float()
            active_update = (~fooled_mask) if not self.continue_after_success else torch.ones(B, dtype=torch.bool, device=device)
            candidate_delta = delta + self.alpha * grad.sign() * support_mask * active_update.view(B, 1, 1, 1).float()

            delta_proj = project_l0(candidate_delta, self.k)
            if not self.continue_after_success:
                delta = torch.where(fooled_mask.view(B, 1, 1, 1), delta, delta_proj)
            else:
                delta = delta_proj
            adv_x_proj = torch.clamp(orig_x + delta, 0.0, 1.0)

            with torch.no_grad():
                out_step = self.model(adv_x_proj)
                forward_evals += 1
                curr_ce = self.ce_loss(out_step, y)
                preds = out_step.argmax(dim=1)
                cand_succ = (preds != y)
                cand_l0 = compute_spatial_l0(adv_x_proj - orig_x).float()

                # Success-First Selection
                replace = (
                    (cand_succ & ~best_succ) |
                    (cand_succ & best_succ & ((cand_l0 < best_l0) | ((cand_l0 == best_l0) & (curr_ce > best_loss)))) |
                    (~cand_succ & ~best_succ & (curr_ce > best_loss))
                )

                best_adv[replace] = adv_x_proj[replace]
                best_succ[replace] = cand_succ[replace]
                best_l0[replace] = cand_l0[replace]
                best_loss[replace] = curr_ce[replace]
                fooled_mask = fooled_mask | cand_succ

        # Iterative Support Pruning post-process (Section 13.1)
        if self.pruning:
            with torch.no_grad():
                for idx in range(B):
                    if not best_succ[idx]:
                        continue
                    single_x = orig_x[idx:idx+1]
                    single_adv = best_adv[idx:idx+1]
                    single_y = y[idx:idx+1]
                    single_delta = single_adv - single_x

                    for pass_num in range(self.pruning_max_passes):
                        channel_mag = single_delta.abs().max(dim=1).values.squeeze(0)
                        active_coords = torch.nonzero(channel_mag > 1e-5)
                        if active_coords.size(0) <= 1:
                            break

                        # Sort active coordinates by magnitude ascending (prune smallest first)
                        mags = channel_mag[active_coords[:, 0], active_coords[:, 1]]
                        sorted_order = torch.argsort(mags)
                        active_coords = active_coords[sorted_order]

                        removed_any = False
                        for coord in active_coords:
                            h, w = coord[0], coord[1]
                            pruned_delta = single_delta.clone()
                            pruned_delta[:, :, h, w] = 0.0
                            pruned_adv = torch.clamp(single_x + pruned_delta, 0.0, 1.0)
                            p_out = self.model(pruned_adv)
                            forward_evals += 1
                            if p_out.argmax(dim=1) != single_y:
                                single_delta = pruned_delta
                                single_adv = pruned_adv
                                removed_any = True

                        if not removed_any:
                            break

                    best_adv[idx:idx+1] = single_adv

        feature_adapter.remove()
        return AttackOutput(
            x_adv=best_adv.detach(),
            forward_evals=forward_evals,
            backward_evals=backward_evals
        )
