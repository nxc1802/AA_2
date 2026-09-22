"""scripts/run_defense_benchmark.py

Benchmark evaluation of sparse adversarial attacks (CASA, SPGD, Sigma-Zero, Sparse-RS)
against standard preprocessing defenses:
- Gaussian Blur (k=3, sigma=1.0)
- Median Filtering (k=3)
- JPEG Compression (Q=75)
- Total Variation Minimization (TVM, 5 iterations)

Evaluates both:
1. Oblivious Attack: Attacker generates perturbations on the undefended model,
   which are then evaluated through the defense filter.
2. Adaptive Attack (BPDA): Attacker utilizes Straight-Through Estimator gradients
   to circumvent non-differentiable preprocessing.
"""

import os
import sys
import json
import argparse
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aa.utils import get_best_device, set_seed
from aa.models import get_model
from aa.data import get_sample_batch_indices
from aa.defenses import (
    GaussianBlurDefense,
    MedianFilterDefense,
    JPEGDefense,
    TVMDefense,
    DefendedModelAdapter,
)
from aa.attacks.casa import CASAAttack
from aa.attacks.external.spgd import SparsePGD
from aa.attacks.external.sigma_zero import SigmaZero


def run_defense_evaluation(
    model,
    dataloader,
    attacks_dict,
    defenses_dict,
    k_values=[4, 16],
    device=None,
    mode="oblivious"
):
    if device is None:
        device = get_best_device()

    results = {}

    for def_name, def_obj in defenses_dict.items():
        print(f"\n==========================================")
        print(f"Evaluating Defense: {def_name} (Mode: {mode})")
        print(f"==========================================")
        results[def_name] = {}

        if def_obj is None:
            defended_model = model
        else:
            defended_model = DefendedModelAdapter(model, defense=def_obj, mode=mode)

        # 1. Clean Accuracy under defense
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                out = defended_model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        clean_acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"[{def_name}] Clean Accuracy: {clean_acc:.2f}% ({correct}/{total})")
        results[def_name]["clean_accuracy"] = clean_acc
        results[def_name]["total_samples"] = total
        results[def_name]["attacks"] = {}

        for atk_name, atk_factory in attacks_dict.items():
            results[def_name]["attacks"][atk_name] = {}
            for k in k_values:
                print(f"  -> Running {atk_name} at K={k}...")
                succ_count = 0
                clean_correct_count = 0
                total_evaluated = 0
                t0 = time.time()

                attack = atk_factory(k=k, model=model if mode == "oblivious" else defended_model)

                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    # Filter clean-correct
                    with torch.no_grad():
                        clean_preds = model(x).argmax(dim=1)
                        mask = clean_preds == y
                        if mask.sum().item() == 0:
                            continue
                        x_clean = x[mask]
                        y_clean = y[mask]
                        clean_correct_count += len(y_clean)

                    # Generate attack
                    out = attack.attack(x_clean, y_clean)
                    x_adv = out.adv_images

                    # Evaluate through defense
                    with torch.no_grad():
                        def_out = defended_model(x_adv)
                        adv_preds = def_out.argmax(dim=1)
                        succ_count += (adv_preds != y_clean).sum().item()
                        total_evaluated += len(y_clean)

                elapsed = time.time() - t0
                asr = 100.0 * succ_count / total_evaluated if total_evaluated > 0 else 0.0
                print(f"     [{atk_name} @ K={k}] ASR: {asr:.2f}% ({succ_count}/{total_evaluated}) in {elapsed:.2f}s")

                results[def_name]["attacks"][atk_name][f"k_{k}"] = {
                    "asr": asr,
                    "success_count": succ_count,
                    "clean_correct_count": total_evaluated,
                    "runtime_seconds": elapsed,
                }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Defense Benchmark for CASA")
    parser.add_argument("--samples", type=int, default=100, help="Number of test samples (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 16], help="K values")
    parser.add_argument("--output", type=str, default="result/defense_benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device) if args.device else get_best_device()
    print(f"Running Defense Benchmark on {device} with {args.samples} samples...")

    # Load Model
    model = get_model("resnet18", num_classes=10, strict_checkpoint=False, checkpoint_path="result/saved_models/resnet18_cifar10_best.pth", device=device)
    model.eval()

    # Load Data
    dataloader, _, _ = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=args.batch_size,
        num_samples=args.samples,
        seed=42
    )

    # Defenses to evaluate
    defenses = {
        "Undefended": None,
        "GaussianBlur (3x3, s=1.0)": GaussianBlurDefense(kernel_size=3, sigma=1.0),
        "MedianFilter (3x3)": MedianFilterDefense(kernel_size=3),
        "JPEGCompression (Q=75)": JPEGDefense(quality=75),
        "TotalVariationMin (5 iters)": TVMDefense(iters=5, step_size=0.05),
    }

    # Attacks to compare
    attacks = {
        "CASA": lambda k, model: CASAAttack(model=model, k=k, steps=20, device=device),
        "SPGD": lambda k, model: SparsePGD(model=model, k=k, steps=20, device=device),
        "SigmaZero": lambda k, model: SigmaZero(model=model, max_k=k, steps=20, device=device),
    }

    results = run_defense_evaluation(
        model=model,
        dataloader=dataloader,
        attacks_dict=attacks,
        defenses_dict=defenses,
        k_values=args.k_values,
        device=device,
        mode="oblivious"
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDefense benchmark saved to {args.output}")


if __name__ == "__main__":
    main()
