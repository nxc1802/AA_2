# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.datasets.dataset_loader import get_sample_batch
from src.models.model_factory import get_model

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS
# ==============================================================================
COALITION_SIZE = 15  # L0 Sparsity Budget
STEPS = 25
ALPHA = 4 / 255.0
COOPERATION_WEIGHT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../result"))
METRICS_DIR = os.path.join(RESULT_DIR, "metrics")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("CPA_Attack")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "cpa_attack.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# PROPOSED METHOD OPTION A: COOPERATIVE PIXELS ATTACK (CPA)
# ==============================================================================
class CooperativePixelsAttack:
    """Option A: Cooperative Pixels Attack (CPA)."""
    def __init__(self, model, coalition_size=COALITION_SIZE, steps=STEPS, alpha=ALPHA, coop_weight=COOPERATION_WEIGHT, device=DEVICE):
        if not isinstance(model, nn.Module):
            raise TypeError("Provided model is not a valid PyTorch nn.Module.")
        self.model = model
        self.coalition_size = coalition_size
        self.steps = steps
        self.alpha = alpha
        self.coop_weight = coop_weight
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, images, labels):
        orig_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        B, C, H, W = orig_images.shape
        adv_images = orig_images.clone().detach()

        kernel = torch.ones(1, 1, 3, 3, device=self.device) / 9.0

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
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()

            if adv_images.grad is None:
                raise RuntimeError("Gradient calculation failed in CPA.")

            grad = adv_images.grad.data
            grad_mag = grad.abs().sum(dim=1)  # (B, H, W)
            local_coop = F.conv2d(grad_mag.unsqueeze(1), kernel, padding=1).squeeze(1)
            coop_score = grad_mag + self.coop_weight * local_coop

            flat_score = coop_score.view(B, -1)
            topk_vals, _ = torch.topk(flat_score, k=min(self.coalition_size, H*W), dim=1)
            k_th_thresh = topk_vals[:, -1].view(B, 1, 1)

            coalition_mask = (coop_score >= k_th_thresh).unsqueeze(1).float()
            
            # Only update active samples
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * coalition_mask * active_mask
            adv_images = torch.clamp(adv_images + step_update, 0.0, 1.0).detach()

            with torch.no_grad():
                preds = self.model(adv_images).argmax(dim=1)
            current_fooled = (preds != labels)
            newly_fooled = current_fooled & (~fooled_mask)
            steps_to_fool[newly_fooled] = step + 1
            fooled_mask = fooled_mask | newly_fooled

        self.last_steps = steps_to_fool.cpu().numpy().tolist()
        return adv_images


def compute_metrics(original, perturbed, orig_labels, adv_preds):
    diff = perturbed - original
    l0 = torch.sum(torch.max(torch.abs(diff), dim=1)[0] > 1e-4, dim=(1, 2)).float().mean().item()
    l2 = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1).mean().item()
    linf = torch.norm(diff.view(diff.size(0), -1), p=float("inf"), dim=1).mean().item()
    asr = (orig_labels != adv_preds).float().mean().item() * 100.0
    return {"ASR (%)": asr, "L0": l0, "L2": l2, "L_inf": linf}

if __name__ == "__main__":
    logger.info("=== Standalone Test: CPA Attack ===")
    model = get_model("resnet18", pretrained=False)
    model.eval()
    loader = get_sample_batch(num_samples=16)
    attacker = CooperativePixelsAttack(model, device=DEVICE)

    for images, labels in loader:
        adv = attacker.attack(images.to(DEVICE), labels.to(DEVICE))
        logger.info(f"CPA output shape: {adv.shape}")
