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
logger = logging.getLogger("VisualizeCAM_FFT")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "visualize_cam_fft.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# SALIENCY MAP & FFT VISUALIZATION LOGIC
# ==============================================================================
def compute_fft_spectrum(tensor_image):
    gray = tensor_image.mean(dim=0)
    fft_c = torch.fft.fftshift(torch.fft.fft2(gray))
    magnitude = torch.log(torch.abs(fft_c) + 1e-8)
    return magnitude.cpu().numpy()

def compute_saliency_map(model, tensor_image, label):
    tensor_image = tensor_image.unsqueeze(0).clone().detach().requires_grad_(True)
    outputs = model(tensor_image)
    loss = outputs[0, label]
    model.zero_grad()
    loss.backward()

    if tensor_image.grad is None:
        raise RuntimeError("Failed to compute saliency map gradient.")

    saliency = tensor_image.grad.data.abs().sum(dim=1).squeeze(0)
    return saliency.cpu().numpy()

def generate_cam_fft_analysis(model, loader, device=DEVICE, save_path=None):
    model = prepare_model_for_eval(model, device)
    images, labels = next(iter(loader))
    img, lbl = images[0].to(device), labels[0].to(device)

    pgd = PGDAttack(model, steps=15, device=device)
    jsma = JSMAAttack(model, k=15, device=device)

    adv_pgd = pgd.attack(img.unsqueeze(0), lbl.unsqueeze(0))[0]
    adv_jsma = jsma.attack(img.unsqueeze(0), lbl.unsqueeze(0))[0]

    samples = [
        ("Original", img),
        ("PGD (Dense)", adv_pgd),
        ("JSMA (Sparse)", adv_jsma)
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for idx, (title, sample_img) in enumerate(samples):
        # 1. Image
        img_np = sample_img.cpu().permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(np.clip(img_np, 0, 1))
        axes[idx, 0].set_title(f"{title} Image")
        axes[idx, 0].axis("off")

        # 2. Input Saliency Map
        sal = compute_saliency_map(model, sample_img, lbl.item())
        sns.heatmap(sal, ax=axes[idx, 1], cmap="hot", cbar=False)
        axes[idx, 1].set_title(f"{title} Saliency Map")
        axes[idx, 1].axis("off")

        # 3. FFT Spectrum
        fft_spec = compute_fft_spectrum(sample_img)
        axes[idx, 2].imshow(fft_spec, cmap="inferno")
        axes[idx, 2].set_title(f"{title} FFT Spectrum")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Input Saliency & FFT visual analysis to {save_path}")
    plt.close()

if __name__ == "__main__":
    logger.info("=== Running Standalone CAM & FFT Visualizer Test ===")
    model = get_model("resnet18", pretrained=False)
    ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if ckpt:
        model = get_model(checkpoint_path=ckpt, device=DEVICE)
    loader = get_sample_batch(batch_size=4, num_samples=4)
    save_file = os.path.join(VIS_DIR, "cam_fft_analysis.png")
    generate_cam_fft_analysis(model, loader, device=DEVICE, save_path=save_file)
    logger.info("=== CAM & FFT Visualizer Complete ===")
