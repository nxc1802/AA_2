import torch
import torch.nn as nn
from src.core.projections import exact_spatial_topk_mask
from src.core.utils import prepare_model_for_eval

class FeatureToMinimalSupportAttack:
    """
    Feature-to-Minimal Support Attack (FMSA).
    Feature Representation -> Minimal Pixel Support Search.
    Disrupts penultimate feature representations relative to clean input features.
    """
    def __init__(self, model: nn.Module, support_budget: int = 15, steps: int = 25, alpha: float = 4/255.0, device: torch.device = None):
        self.model = prepare_model_for_eval(model, device)
        self.support_budget = support_budget
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.extracted_features = None
        self.hook_handle = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        """Hook penultimate layer output to track feature representation."""
        def hook(module, input, output):
            self.extracted_features = output
        
        if hasattr(self.model, "layer4"):
            self.hook_handle = self.model.layer4.register_forward_hook(hook)

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __del__(self):
        self.remove_hook()

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        adv_images = orig_images.clone().detach()

        # Extract clean baseline feature representation
        with torch.no_grad():
            self.model(orig_images)
            clean_features = self.extracted_features.clone().detach() if self.extracted_features is not None else None

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        
        with torch.no_grad():
            preds = self.model(adv_images).argmax(dim=1)
        fooled_mask = (preds != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            if self.extracted_features is not None and clean_features is not None:
                # Maximize L2 distance from clean feature representation
                feat_dist = (self.extracted_features - clean_features).pow(2).sum(dim=[1, 2, 3])
                crit_loss = feat_dist.sum()
            else:
                crit_loss = nn.CrossEntropyLoss()(outputs, labels)

            self.model.zero_grad()
            crit_loss.backward()

            grad = adv_images.grad.data
            grad_mag = grad.abs().sum(dim=1, keepdim=True) # (B, 1, H, W)

            support_mask = exact_spatial_topk_mask(grad_mag, self.support_budget).float()
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            
            step_update = self.alpha * grad.sign() * support_mask * active_mask
            adv_images = torch.clamp(adv_images + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
                current_fooled = (preds != labels)
                newly_fooled = current_fooled & (~fooled_mask)
                steps_to_fool[newly_fooled] = step + 1
                fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return adv_images
