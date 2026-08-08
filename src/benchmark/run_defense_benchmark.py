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
from src.core.utils import prepare_model_for_eval, get_best_device

# Defenses & Adaptive BPDA Adapter
from src.defenses.preprocessing.gaussian_blur import GaussianBlurDefense
from src.defenses.preprocessing.median_filter import MedianFilterDefense
from src.defenses.preprocessing.jpeg_compression import JPEGCompressionDefense
from src.defenses.preprocessing.tvm import TotalVariationMinimizationDefense
from src.defenses.bpda_adapter import DefendedModelAdapter

# Benchmark Attacks
from src.attacks.baselines.pgd import PGDAttack
from src.attacks.classical.jsma import JSMAAttack
from src.attacks.optimization.pgd0 import PGD0Attack
from src.attacks.proposed.cpa import CooperativePixelsAttack

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS
# ==============================================================================
NUM_SAMPLES = 50
BATCH_SIZE = 64
DEVICE = get_best_device()
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
# ADAPTIVE & OBLIVIOUS BENCHMARK ENGINE
# ==============================================================================
def evaluate_defenses(model, loader, eval_mode="adaptive", device=DEVICE):
    """
    Evaluates defenses under Adaptive BPDA (g(x) = f(D(x))) or Oblivious (attack generated against f(x)).
    Reports percentage-point Recovery Delta (pp) and Relative Recovery (%).
    """
    model = prepare_model_for_eval(model, device)

    defenses = {
        "Gaussian Blur": GaussianBlurDefense(kernel_size=3, sigma=1.0),
        "Median Filter": MedianFilterDefense(kernel_size=3),
        "JPEG (Q=75)": JPEGCompressionDefense(quality=75),
        "TVM": TotalVariationMinimizationDefense(steps=10, device=device),
    }

    clean_batches = []
    labels_batches = []
    for images, labels in loader:
        clean_batches.append(images.to(device))
        labels_batches.append(labels.to(device))
    clean_images = torch.cat(clean_batches, dim=0)
    all_labels = torch.cat(labels_batches, dim=0)

    # Compute baseline clean accuracy without defense
    with torch.no_grad():
        clean_preds = torch.argmax(model(clean_images), dim=1)
        clean_acc_base = 100.0 * (clean_preds == all_labels).float().mean().item()

    # Pre-compute clean utility for each defense
    clean_acc_defended = {}
    for def_name, defense in defenses.items():
        def_clean_imgs = defense.defend(clean_images)
        with torch.no_grad():
            def_clean_preds = torch.argmax(model(def_clean_imgs), dim=1)
            clean_acc_defended[def_name] = 100.0 * (def_clean_preds == all_labels).float().mean().item()

    results = []

    for def_name, defense in defenses.items():
        logger.info(f"=== Evaluating Defense: {def_name} (Mode: {eval_mode.upper()}) ===")
        
        # Defended Model Adapter (BPDA for non-differentiable defenses, differentiable for GaussianBlur)
        defended_target_model = DefendedModelAdapter(model, defense=defense, mode=eval_mode)

        test_attacks = {
            "PGD (Dense)": PGDAttack(defended_target_model, steps=10, device=device),
            "JSMA (Sparse)": JSMAAttack(defended_target_model, k=15, device=device),
            "PGD0 (Opt Sparse)": PGD0Attack(defended_target_model, k=15, steps=15, device=device),
            "CPA (Proposed)": CooperativePixelsAttack(defended_target_model, coalition_size=15, steps=15, device=device)
        }

        for atk_name, attacker in test_attacks.items():
            adv_batches = []
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                adv = attacker.attack(images, labels)
                adv_batches.append(adv)

            adv_images = torch.cat(adv_batches, dim=0)

            # Measure undefended accuracy vs defended accuracy
            with torch.no_grad():
                undefended_preds = torch.argmax(model(adv_images), dim=1)
                undef_acc = 100.0 * (undefended_preds == all_labels).float().mean().item()

                defended_imgs = defense.defend(adv_images)
                defended_preds = torch.argmax(model(defended_imgs), dim=1)
                def_acc = 100.0 * (defended_preds == all_labels).float().mean().item()

            rec_delta_pp = def_acc - undef_acc
            rel_rec_pct = 100.0 * rec_delta_pp / max(1.0, 100.0 - undef_acc)

            res = {
                "Defense": def_name,
                "Evaluation Mode": eval_mode,
                "Attack": atk_name,
                "Clean Acc Undefended (%)": round(clean_acc_base, 2),
                "Clean Acc Defended (%)": round(clean_acc_defended[def_name], 2),
                "Undefended Acc (%)": round(undef_acc, 2),
                "Defended Acc (%)": round(def_acc, 2),
                "Recovery Delta (pp)": round(rec_delta_pp, 2),
                "Relative Recovery (%)": round(rel_rec_pct, 2)
            }
            logger.info(f"Result: {res}")
            results.append(res)

    return pd.DataFrame(results)

if __name__ == "__main__":
    logger.info("=== Running High-Performance Defense Benchmark (Adaptive BPDA & Oblivious) ===")
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else NUM_SAMPLES
    allow_untrained = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    eval_mode = sys.argv[3].lower() if len(sys.argv) > 3 else "adaptive"
    
    ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if not ckpt and not allow_untrained:
        raise FileNotFoundError(
            "Model checkpoint 'resnet18_cifar10_best.pth' not found! "
            "Benchmarking defenses on an untrained model is scientifically invalid. "
            "Please train or download checkpoint, or pass allow_untrained=True to force."
        )
    model = get_model("resnet18", pretrained=False, checkpoint_path=ckpt if ckpt else None, device=DEVICE)
        
    loader = get_sample_batch(batch_size=BATCH_SIZE, num_samples=num_samples)

    df_results = evaluate_defenses(model, loader, eval_mode=eval_mode, device=DEVICE)

    csv_file = os.path.join(METRICS_DIR, f"full_defense_benchmark_{eval_mode}.csv")
    json_file = os.path.join(METRICS_DIR, f"full_defense_benchmark_{eval_mode}.json")

    df_results.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=4)

    logger.info(f"\n{df_results.to_string(index=False)}")
    logger.info(f"Saved defense benchmark metrics to {csv_file} and {json_file}")
