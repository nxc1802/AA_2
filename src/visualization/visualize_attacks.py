# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core import prepare_model_for_eval
from src.datasets.dataset_loader import get_sample_batch
from src.models.model_factory import get_model, find_existing_checkpoint
from src.attacks.baselines.pgd import PGDAttack
from src.attacks.classical.jsma import JSMAAttack
from src.attacks.optimization.pgd0 import PGD0Attack

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../result"))
VIS_DIR = os.path.join(RESULT_DIR, "visualizations")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("VisualizeAttacks")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "visualize_attacks.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================
def visualize_sample_attacks(model, loader, device=DEVICE, save_path=None):
    """Generates comparison grid: Original | PGD (Dense) | JSMA (Sparse) | PGD0."""
    model = prepare_model_for_eval(model, device)
    images, labels = next(iter(loader))
    images, labels = images[:3].to(device), labels[:3].to(device)

    attacks = {
        "PGD (Dense)": PGDAttack(model, steps=15, device=device),
        "JSMA (Classical)": JSMAAttack(model, k=15, device=device),
        "PGD0 (Opt)": PGD0Attack(model, k=15, steps=15, device=device),
    }

    num_samples = images.size(0)
    fig, axes = plt.subplots(num_samples, len(attacks) + 1, figsize=(14, 3 * num_samples))

    for row in range(num_samples):
        orig_img = images[row].cpu().permute(1, 2, 0).numpy()
        axes[row, 0].imshow(np.clip(orig_img, 0, 1))
        axes[row, 0].set_title(f"Original (L:{labels[row].item()})")
        axes[row, 0].axis("off")

        col = 1
        for name, attacker in attacks.items():
            adv_img_t = attacker.attack(images[row:row+1], labels[row:row+1])
            adv_img = adv_img_t[0].cpu().permute(1, 2, 0).numpy()

            with torch.no_grad():
                pred = torch.argmax(model(adv_img_t), dim=1).item()

            axes[row, col].imshow(np.clip(adv_img, 0, 1))
            axes[row, col].set_title(f"{name}\nPred: {pred}")
            axes[row, col].axis("off")
            col += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved attack visualization grid to {save_path}")
    plt.close()

if __name__ == "__main__":
    logger.info("=== Running Standalone Attack Visualizer Test ===")
    model = get_model("resnet18", pretrained=False)
    ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if ckpt:
        model = get_model(checkpoint_path=ckpt, device=DEVICE)
    loader = get_sample_batch(batch_size=8, num_samples=8)
    save_file = os.path.join(VIS_DIR, "attack_comparison_grid.png")
    visualize_sample_attacks(model, loader, device=DEVICE, save_path=save_file)
    logger.info("=== Attack Visualizer Complete ===")
