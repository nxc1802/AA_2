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
HYPERGRAPH_BUDGET = 15
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
logger = logging.getLogger("HSA_Attack")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "hsa_attack.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# PROPOSED METHOD OPTION D: HYPERGRAPH SPARSE ATTACK (HSA)
# ==============================================================================
class HypergraphSparseAttack:
    """
    Option D: Hypergraph Sparse Attack (HSA).
    Constructs a Hypergraph structure (Nodes = pixels, Hyperedges = representation feature channels).
    Performs minimum coalition search to break maximum hyperedges.
    """
    def __init__(self, model, budget=HYPERGRAPH_BUDGET, steps=STEPS, alpha=ALPHA, device=DEVICE):
        if not isinstance(model, nn.Module):
            raise TypeError("Provided model is not a valid PyTorch nn.Module.")
        self.model = model
        self.budget = budget
        self.steps = steps
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def _construct_hypergraph_degree(self, images, labels):
        """Constructs hypergraph incidence matrix and calculates node degree (pixel criticality across hyperedges)."""
        images.requires_grad = True
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.model.zero_grad()
        loss.backward()

        if images.grad is None:
            raise RuntimeError("Gradient computation failed during HSA hypergraph construction.")

        grad = images.grad.data
        grad_mag = grad.abs().sum(dim=1)  # Node importance across spatial grid (B, H, W)

        # Hyperedge degree aggregation: pooling across multi-scale spatial receptive fields
        hyperedge_pool1 = F.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)
        hyperedge_pool2 = F.avg_pool2d(grad_mag, kernel_size=5, stride=1, padding=2)

        # Total Hypergraph Node Centrality = Base Pixel Gradient + Combined Hyperedge Degree
        node_centrality = grad_mag + hyperedge_pool1 + hyperedge_pool2
        return node_centrality, grad

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

            node_centrality, grad = self._construct_hypergraph_degree(adv_images, labels)

            # Coalition search: find min pixels breaking max hyperedges
            flat_centrality = node_centrality.view(B, -1)
            topk_vals, _ = torch.topk(flat_centrality, k=min(self.budget, H*W), dim=1)
            thresh = topk_vals[:, -1].view(B, 1, 1)

            hypergraph_mask = (node_centrality >= thresh).unsqueeze(1).float()
            
            # Only update active samples
            active_mask = (~fooled_mask).view(B, 1, 1, 1).float()
            step_update = self.alpha * grad.sign() * hypergraph_mask * active_mask
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
    logger.info("=== Standalone Test: HSA Attack ===")
    model = get_model("resnet18", pretrained=False)
    model.eval()
    loader = get_sample_batch(num_samples=16)
    attacker = HypergraphSparseAttack(model, device=DEVICE)

    for images, labels in loader:
        adv = attacker.attack(images.to(DEVICE), labels.to(DEVICE))
        logger.info(f"HSA output shape: {adv.shape}")
