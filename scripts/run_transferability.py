"""scripts/run_transferability.py

Evaluates Black-Box Cross-Model Transferability of Sparse Adversarial Perturbations.
Generates adversarial perturbations on a Source Model (e.g., ResNet-18), and evaluates
whether those sparse pixel modifications fool Target Models without access to their gradients:
- Source: ResNet-18
- Targets: WideResNet-28-10, ResNet-50, MobileNetV2, ViTCIFAR

Computes Transfer Attack Success Rate:
T-ASR(Ms -> Mt) = P( Mt(x + delta) != y | Ms(x) = y AND Mt(x) = y )
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
from aa.attacks.casa import CASAAttack
from aa.attacks.external.spgd import SparsePGD


def run_transfer_evaluation(
    source_model,
    target_models_dict,
    dataloader,
    attacks_dict,
    k_values=[1, 4, 16],
    device=None
):
    if device is None:
        device = get_best_device()

    results = {
        "source_model": getattr(source_model, "architecture_name", "source"),
        "target_models": list(target_models_dict.keys()),
        "k_values": k_values,
        "transfer_matrix": {}
    }

    for atk_name, atk_factory in attacks_dict.items():
        print(f"\n==========================================")
        print(f"Generating Transfer Attacks using: {atk_name}")
        print(f"==========================================")
        results["transfer_matrix"][atk_name] = {}

        for k in k_values:
            print(f"  -> Budget K={k}...")
            results["transfer_matrix"][atk_name][f"k_{k}"] = {}
            attack = atk_factory(k=k, model=source_model)

            # Store adversarial samples and clean labels
            adv_samples = []
            clean_labels = []
            clean_images = []

            t0 = time.time()
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    # Filter clean correct on source model
                    src_preds = source_model(x).argmax(dim=1)
                    mask = src_preds == y
                    if mask.sum().item() == 0:
                        continue
                    x_sub = x[mask]
                    y_sub = y[mask]

                out = attack.attack(x_sub, y_sub)
                adv_samples.append(out.x_adv.detach().cpu())
                clean_labels.append(y_sub.detach().cpu())
                clean_images.append(x_sub.detach().cpu())

            gen_time = time.time() - t0
            print(f"     Generated perturbations in {gen_time:.2f}s")

            if len(adv_samples) == 0:
                continue

            all_x_adv = torch.cat(adv_samples, dim=0)
            all_x_clean = torch.cat(clean_images, dim=0)
            all_y = torch.cat(clean_labels, dim=0)
            N = len(all_y)

            for tgt_name, tgt_model in target_models_dict.items():
                tgt_model.eval()
                succ_transfer = 0
                joint_clean_correct = 0

                batch_size = 64
                with torch.no_grad():
                    for i in range(0, N, batch_size):
                        bx_clean = all_x_clean[i:i+batch_size].to(device)
                        bx_adv = all_x_adv[i:i+batch_size].to(device)
                        by = all_y[i:i+batch_size].to(device)

                        # Clean correct on target
                        tgt_clean_pred = tgt_model(bx_clean).argmax(dim=1)
                        joint_mask = tgt_clean_pred == by
                        joint_count = joint_mask.sum().item()
                        if joint_count == 0:
                            continue
                        joint_clean_correct += joint_count

                        # Adv pred on target
                        tgt_adv_pred = tgt_model(bx_adv[joint_mask]).argmax(dim=1)
                        succ_transfer += (tgt_adv_pred != by[joint_mask]).sum().item()

                t_asr = 100.0 * succ_transfer / joint_clean_correct if joint_clean_correct > 0 else 0.0
                print(f"     [Target: {tgt_name}] T-ASR: {t_asr:.2f}% ({succ_transfer}/{joint_clean_correct})")

                results["transfer_matrix"][atk_name][f"k_{k}"][tgt_name] = {
                    "t_asr": t_asr,
                    "succ_transfer": succ_transfer,
                    "joint_clean_correct": joint_clean_correct,
                }

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Transferability Benchmark")
    parser.add_argument("--samples", type=int, default=100, help="Test samples (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 4, 16], help="Budget K values")
    parser.add_argument("--output", type=str, default="result/transferability_results.json", help="Output path")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device) if args.device else get_best_device()
    print(f"Running Cross-Model Transferability on {device} ({args.samples} samples)...")

    # Load Source Model (ResNet-18)
    source_model = get_model("resnet18", num_classes=10, strict_checkpoint=False, checkpoint_path="result/saved_models/resnet18_cifar10_best.pth", device=device)
    source_model.eval()

    # Load Target Models
    target_models = {
        "ResNet-18 (White-Box)": source_model,
        "WideResNet-28-10": get_model("wideresnet28_10", num_classes=10, strict_checkpoint=False, device=device),
        "ResNet-50": get_model("resnet50", num_classes=10, strict_checkpoint=False, device=device),
        "MobileNetV2": get_model("mobilenet_v2", num_classes=10, strict_checkpoint=False, device=device),
        "ViTCIFAR": get_model("vit_cifar", num_classes=10, strict_checkpoint=False, device=device),
    }

    # Load Data
    dataloader, _, _ = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=args.batch_size,
        num_samples=args.samples,
        seed=42
    )

    attacks = {
        "CASA": lambda k, model: CASAAttack(model=model, k=k, steps=20),
        "SPGD": lambda k, model: SparsePGD(model=model, k=k, steps=20),
    }

    results = run_transfer_evaluation(
        source_model=source_model,
        target_models_dict=target_models,
        dataloader=dataloader,
        attacks_dict=attacks,
        k_values=args.k_values,
        device=device
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTransferability benchmark saved to {args.output}")


if __name__ == "__main__":
    main()
