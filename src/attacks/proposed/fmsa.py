import torch
import torch.nn as nn
from typing import Optional
from src.core.projections import project_l0, exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval

class FeatureExtractorAdapter:
    """Flexible feature extractor adapter for diverse model backbones."""
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


class FeatureToMinimalSupportAttack:
    """
    Feature-to-Minimal Support Attack (FMSA).
    Jointly optimizes classification loss and feature representation disruption:
    L = L_CE + lambda * L_feat
    Support budget is projected strictly onto K-sparse L0 ball.
    """
    def __init__(
        self, 
        model: nn.Module, 
        support_budget: int = 15, 
        steps: int = 25, 
        alpha: float = 4/255.0, 
        feature_weight: float = 1.0,
        layer_name: Optional[str] = None,
        device: torch.device = None
    ):
        self.model = prepare_model_for_eval(model, device)
        self.support_budget = support_budget
        self.steps = steps
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_adapter = FeatureExtractorAdapter(self.model, layer_name=layer_name)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hook()

    def remove_hook(self):
        self.feature_adapter.remove()

    def __del__(self):
        self.remove_hook()

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        delta = torch.zeros_like(orig_images)
        best_adv = orig_images.clone()

        with torch.no_grad():
            out_init = self.model(orig_images)
            clean_features = self.feature_adapter.extracted_features.clone().detach() if self.feature_adapter.extracted_features is not None else None
            best_loss = self.ce_loss(out_init, labels)

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = (out_init.argmax(dim=1) != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            adv_images = (orig_images + delta).clamp(0.0, 1.0).requires_grad_(True)
            outputs = self.model(adv_images)
            
            ce_l = self.ce_loss(outputs, labels)
            
            if self.feature_adapter.extracted_features is not None and clean_features is not None:
                feat_diff = self.feature_adapter.extracted_features - clean_features
                feat_l = feat_diff.pow(2).flatten(1).sum(dim=1)
                total_loss = ce_l + self.feature_weight * feat_l
            else:
                total_loss = ce_l

            loss = total_loss.sum()
            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            grad_mag = grad.abs().sum(dim=1, keepdim=True)
            
            support_mask = exact_spatial_topk_mask(grad_mag, self.support_budget).float()
            candidate_delta = delta + self.alpha * grad.sign() * support_mask
            
            delta = project_l0(candidate_delta, self.support_budget)
            adv_images_proj = torch.clamp(orig_images + delta, 0.0, 1.0)

            with torch.no_grad():
                out_step = self.model(adv_images_proj)
                curr_ce = self.ce_loss(out_step, labels)
                preds = out_step.argmax(dim=1)

                improved = curr_ce > best_loss
                best_loss[improved] = curr_ce[improved]
                best_adv[improved] = adv_images_proj[improved]

                current_fooled = (preds != labels)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        steps_list = steps_to_fool.cpu().numpy().tolist()
        self.last_steps = steps_list
        self.last_queries = [int(s * 2 + 1) for s in steps_list]
        self.last_grad_evals = [int(s) for s in steps_list]
        return best_adv
