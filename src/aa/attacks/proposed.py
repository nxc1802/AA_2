import torch
import torch.nn as nn
from typing import Optional

from aa.attacks.base import Attack, AttackOutput
from aa.metrics import project_l0, exact_spatial_topk_mask, compute_spatial_l0


class FeatureExtractorAdapter:
    """Extracts internal intermediate features for feature guidance."""
    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name
        self.extracted_features = None
        self.hook_handle = None
        self._attach_hook()

    def _attach_hook(self):
        target_layer = None
        if self.layer_name and hasattr(self.model, self.layer_name):
            target_layer = getattr(self.model, self.layer_name)
        elif hasattr(self.model, "layer4"):
            target_layer = self.model.layer4
        elif hasattr(self.model, "features"):
            target_layer = self.model.features

        if target_layer is not None:
            def hook(module, input, output):
                self.extracted_features = output
            self.hook_handle = target_layer.register_forward_hook(hook)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class SparseFeatureAttack(Attack):
    """
    Proposed Method: Feature-Guided Sparse Adversarial Attack with Support Pruning.
    
    Ablation flags:
    - feature_guidance: Add intermediate representation disruption loss.
    - interaction: Add local spatial interaction weighting.
    - pruning: Perform post-success support pruning to minimal L0.
    """
    def __init__(
        self,
        model: nn.Module,
        k: int = 16,
        steps: int = 25,
        alpha: float = 4 / 255.0,
        feature_weight: float = 1.0,
        feature_guidance: bool = True,
        interaction: bool = False,
        pruning: bool = True,
        layer_name: Optional[str] = None,
    ):
        self.model = model
        self.k = k
        self.steps = steps
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.feature_guidance = feature_guidance
        self.interaction = interaction
        self.pruning = pruning
        self.layer_name = layer_name
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> AttackOutput:
        device = x.device
        B, C, H, W = x.shape
        orig_x = x.clone().detach()
        y = y.clone().detach()

        feature_adapter = FeatureExtractorAdapter(self.model, layer_name=self.layer_name)

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

            if self.feature_guidance and feature_adapter.extracted_features is not None and clean_features is not None:
                feat_diff = feature_adapter.extracted_features - clean_features
                feat_l = feat_diff.pow(2).flatten(1).sum(dim=1)
                total_loss = ce_l + self.feature_weight * feat_l
            else:
                total_loss = ce_l

            loss = total_loss.sum()
            self.model.zero_grad()
            loss.backward()
            backward_evals += 1

            grad = adv_x.grad.data
            grad_mag = grad.abs().sum(dim=1, keepdim=True)

            if self.interaction:
                # Add local spatial smoothing as interaction score
                grad_mag = grad_mag + 0.5 * torch.nn.functional.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)

            grad_mag_masked = grad_mag.clone()
            grad_mag_masked[fooled_mask] = -float("inf")

            support_mask = exact_spatial_topk_mask(grad_mag_masked, self.k).float()
            candidate_delta = delta + self.alpha * grad.sign() * support_mask * (~fooled_mask).view(B, 1, 1, 1).float()

            delta_proj = project_l0(candidate_delta, self.k)
            delta = torch.where(fooled_mask.view(B, 1, 1, 1), delta, delta_proj)
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

        # Optional Support Pruning post-process
        if self.pruning:
            with torch.no_grad():
                for idx in range(B):
                    if not best_succ[idx]:
                        continue
                    single_x = orig_x[idx:idx+1]
                    single_adv = best_adv[idx:idx+1]
                    single_y = y[idx:idx+1]
                    single_delta = single_adv - single_x

                    # Find active spatial locations
                    channel_mag = single_delta.abs().max(dim=1).values.squeeze(0) # (H, W)
                    active_coords = torch.nonzero(channel_mag > 1e-5) # (N, 2)

                    if active_coords.size(0) > 1:
                        # Try removing active pixels one by one
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
                        best_adv[idx:idx+1] = single_adv

        feature_adapter.remove()
        return AttackOutput(
            x_adv=best_adv.detach(),
            forward_evals=forward_evals,
            backward_evals=backward_evals
        )
