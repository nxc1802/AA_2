# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import logging
import json
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.datasets.dataset_loader import get_sample_batch
from src.models.model_factory import get_model

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS
# ==============================================================================
MINIMAL_SUPPORT_K = 15
STEPS = 25
ALPHA = 4 / 255.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../result"))
METRICS_DIR = os.path.join(RESULT_DIR, "metrics")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("FMSA_Attack")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "fmsa_attack.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# PROPOSED METHOD OPTION C: FEATURE-TO-MINIMAL SUPPORT ATTACK (FMSA)
# ==============================================================================
class FeatureToMinimalSupportAttack:
    """
    Option C: Feature-to-Minimal Support Attack (FMSA).
    Reverses standard pipeline: Feature Representation -> Minimal Pixel Support Search.
    Selects critical penultimate features and computes minimal pixel support that collapses them.
    """
    def __init__(self, model, support_budget=MINIMAL_SUPPORT_K, steps=STEPS, alpha=ALPHA, device=DEVICE):
        if not isinstance(model, nn.Module):
            raise TypeError("Provided model is not a valid PyTorch nn.Module.")
        self.model = model
        self.support_budget = support_budget
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extracted_features = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        """Hook penultimate layer output to track feature representation."""
        def hook(module, input, output):
            self.extracted_features = output
        
        # Register on layer4 (penultimate conv layer of ResNet)
        if hasattr(self.model, "layer4"):
            self.model.layer4.register_forward_hook(hook)

    def attack(self, images, labels):
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        adv_images = orig_images.clone().detach()

        steps_to_fool = torch.full((B,), self.steps, dtype=torch.float, device=self.device)
        fooled_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            preds = self.model(adv_images).argmax(dim=1)
        fooled_mask = (preds != labels)
        steps_to_fool[fooled_mask] = 0.0

        for step in range(self.steps):
            if fooled_mask.all():
                break

            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Feature collapse objective: minimize similarity / maximize distance of critical representation
            if self.extracted_features is not None:
                feat = self.extracted_features
                crit_loss = -torch.mean(feat.abs())  # Feature collapse
            else:
                crit_loss = -nn.CrossEntropyLoss()(outputs, labels)

            self.model.zero_grad()
            crit_loss.backward()

            if adv_images.grad is None:
                raise RuntimeError(f"Gradient computation failed at FMSA step {step+1}.")

            grad = adv_images.grad.data
            grad_mag = grad.abs().sum(dim=1)  # (B, H, W)

            # Minimal support search: select top support_budget pixels
            flat_grad = grad_mag.view(B, -1)
            topk_vals, _ = torch.topk(flat_grad, k=min(self.support_budget, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            support_mask = (grad_mag >= thresh).unsqueeze(1).float()
            
            # Only update active samples
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


if __name__ == "__main__":
    logger.info("=== Standalone Test: FMSA Attack ===")
    model = get_model("resnet18", pretrained=False)
    model.eval()
    loader = get_sample_batch(num_samples=16)
    attacker = FeatureToMinimalSupportAttack(model, device=DEVICE)

    for images, labels in loader:
        adv = attacker.attack(images.to(DEVICE), labels.to(DEVICE))
        logger.info(f"FMSA output shape: {adv.shape}")
