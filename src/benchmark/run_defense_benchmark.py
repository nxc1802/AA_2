# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import logging
import json
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.datasets.dataset_loader import get_sample_batch
from src.models.model_factory import get_model, find_existing_checkpoint
from src.core.utils import prepare_model_for_eval

# Defenses
from src.defenses.preprocessing.gaussian_blur import GaussianBlurDefense
from src.defenses.preprocessing.median_filter import MedianFilterDefense
from src.defenses.preprocessing.jpeg_compression import JPEGCompressionDefense
from src.defenses.preprocessing.tvm import TotalVariationMinimizationDefense

# Benchmark Attacks
from src.attacks.baselines.pgd import PGDAttack
from src.attacks.classical.jsma import JSMAAttack
from src.attacks.optimization.pgd0 import PGD0Attack

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS
# ==============================================================================
NUM_SAMPLES = 50
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../result"))
METRICS_DIR = os.path.join(RESULT_DIR, "metrics")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("DefenseBenchmark")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "defense_benchmark.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# BENCHMARK ENGINE
# ==============================================================================
def evaluate_defenses(model, loader, device=DEVICE):
    """Evaluates defense recovery rates against representative dense and sparse attacks."""
    model = prepare_model_for_eval(model, device)

    defenses = {
        "Gaussian Blur": GaussianBlurDefense(kernel_size=3, sigma=1.0),
        "Median Filter": MedianFilterDefense(kernel_size=3),
        "JPEG (Q=75)": JPEGCompressionDefense(quality=75),
        "TVM": TotalVariationMinimizationDefense(steps=10, device=device),
    }

    test_attacks = {
        "PGD (Dense)": PGDAttack(model, steps=10, device=device),
        "JSMA (Classical Sparse)": JSMAAttack(model, k=15, device=device),
        "PGD0 (Opt Sparse)": PGD0Attack(model, k=15, steps=15, device=device),
    }

    results = []

    for atk_name, attacker in test_attacks.items():
        logger.info(f"--- Testing Defenses Against Attack: {atk_name} ---")

        adv_batches = []
        orig_labels = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            adv = attacker.attack(images, labels)
            adv_batches.append(adv)
            orig_labels.append(labels)

        adv_images = torch.cat(adv_batches, dim=0)
        labels = torch.cat(orig_labels, dim=0)

        with torch.no_grad():
            undefended_preds = torch.argmax(model(adv_images), dim=1)
            undef_acc = 100.0 * (undefended_preds == labels).float().mean().item()

        for def_name, defense in defenses.items():
            defended_imgs = defense.defend(adv_images)
            with torch.no_grad():
                defended_preds = torch.argmax(model(defended_imgs), dim=1)
                def_acc = 100.0 * (defended_preds == labels).float().mean().item()

            rec_rate = def_acc - undef_acc
            res = {
                "Attack": atk_name,
                "Defense": def_name,
                "Undefended Acc (%)": round(undef_acc, 2),
                "Defended Acc (%)": round(def_acc, 2),
                "Recovery Rate (%)": round(rec_rate, 2)
            }
            logger.info(f"Result: {res}")
            results.append(res)

    return pd.DataFrame(results)

if __name__ == "__main__":
    logger.info("=== Running High-Performance Defense Benchmark ===")
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else NUM_SAMPLES
    
    model = get_model("resnet18", pretrained=False)
    ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if ckpt:
        model = get_model(checkpoint_path=ckpt, device=DEVICE)
        
    loader = get_sample_batch(batch_size=BATCH_SIZE, num_samples=num_samples)

    df_results = evaluate_defenses(model, loader, device=DEVICE)

    csv_file = os.path.join(METRICS_DIR, "full_defense_benchmark.csv")
    json_file = os.path.join(METRICS_DIR, "full_defense_benchmark.json")

    df_results.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=4)

    logger.info(f"\n{df_results.to_string(index=False)}")
    logger.info(f"Saved defense benchmark metrics to {csv_file} and {json_file}")
