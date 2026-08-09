import argparse
import os
import json
import time
import torch
import numpy as np
from aa.utils import set_seed, get_best_device
from aa.data import get_sample_batch_indices
from aa.models import get_model
from aa.attacks.external.cornersearch import CornerSearch
from aa.benchmark import evaluate_attack, derive_progressive_asr_curve


def main():
    parser = argparse.ArgumentParser(description="Validate CornerSearch Kmax-only progressive single-pass equivalence")
    parser.add_argument("--samples", type=int, default=100, help="Number of CIFAR-10 samples for validation (100-200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="result/cornersearch_kmax_validation.json", help="Path to save output report")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_best_device()

    print(f"=== Running CornerSearch Kmax Equivalence Validation ({args.samples} samples, seed={args.seed}) ===", flush=True)

    loader, sample_indices, sample_hash = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=64,
        num_samples=args.samples,
        seed=args.seed
    )

    model = get_model(
        model_name="resnet18",
        checkpoint_path="resnet18_cifar10_best.pth",
        device=device
    )

    k_values = [1, 2, 4, 8, 16, 32, 64]
    max_k = max(k_values)

    # -------------------------------------------------------------
    # Protocol A: Independent multi-run sweep over K=1..64
    # -------------------------------------------------------------
    print("\n[Protocol A] Sweeping CornerSearch independently for K in [1, 2, 4, 8, 16, 32, 64]...", flush=True)
    sweep_results = {}
    t0_sweep = time.time()
    for k in k_values:
        print(f"  --> Running CornerSearch K={k}...", flush=True)
        atk = CornerSearch(model=model, k=k, max_iter=1000, n_max=100, seed=args.seed)
        res = evaluate_attack(model, atk, loader, device=device)
        sweep_results[f"k_{k}"] = {
            "asr": res["asr"],
            "success_count": res["success_count"],
            "clean_correct_count": res["clean_correct_count"],
            "runtime": res["runtime_seconds"],
            "mean_l0": float(np.mean(res["raw_l0"])) if res["raw_l0"] else 0.0
        }
        print(f"      ASR@{k}: {res['asr']:.2f}%, runtime: {res['runtime_seconds']:.2f}s", flush=True)
    t_sweep_total = time.time() - t0_sweep

    # -------------------------------------------------------------
    # Protocol B: Single-pass Kmax=64 run + derived ASR@K curve
    # -------------------------------------------------------------
    print(f"\n[Protocol B] Single-pass CornerSearch Kmax={max_k} + derived curve...", flush=True)
    t0_single = time.time()
    atk_single = CornerSearch(model=model, k=max_k, max_iter=1000, n_max=100, seed=args.seed)
    base_res = evaluate_attack(model, atk_single, loader, device=device)
    derived_curve = derive_progressive_asr_curve(base_res, k_values)
    t_single_total = time.time() - t0_single

    progressive_results = {}
    prev_asr = -1.0
    is_monotonic = True

    for k in k_values:
        k_res = derived_curve[f"k_{k}"]
        k_asr = k_res["asr"]
        if k_asr < prev_asr - 1e-5:
            is_monotonic = False
        prev_asr = k_asr

        progressive_results[f"k_{k}"] = {
            "asr": k_asr,
            "success_count": k_res["success_count"],
            "clean_correct_count": k_res["clean_correct_count"],
        }

    # Summary table output
    print("\n" + "=" * 60)
    print(f"{'K':<5} | {'Protocol A (Sweep ASR)':<22} | {'Protocol B (Single Pass ASR)':<25}")
    print("-" * 60)
    diffs = {}
    for k in k_values:
        asr_a = sweep_results[f"k_{k}"]["asr"]
        asr_b = progressive_results[f"k_{k}"]["asr"]
        diff = asr_b - asr_a
        diffs[f"k_{k}"] = round(diff, 2)
        print(f"{k:<5} | {asr_a:>21.2f}% | {asr_b:>24.2f}% (diff: {diff:+.2f}%)")
    print("=" * 60)

    speedup = t_sweep_total / t_single_total if t_single_total > 0 else 1.0
    print(f"Monotonicity (Protocol B): {'PASS' if is_monotonic else 'FAIL'}")
    print(f"Total Wall Time: Protocol A = {t_sweep_total:.1f}s | Protocol B = {t_single_total:.1f}s (Speedup: {speedup:.2f}x)")

    report = {
        "metadata": {
            "samples": args.samples,
            "seed": args.seed,
            "sample_indices_hash": sample_hash,
            "device": str(device)
        },
        "sweep_protocol_a": {
            "total_runtime_seconds": t_sweep_total,
            "results": sweep_results
        },
        "progressive_protocol_b": {
            "total_runtime_seconds": t_single_total,
            "is_monotonic": is_monotonic,
            "results": progressive_results
        },
        "diff_b_minus_a": diffs,
        "speedup": round(speedup, 2)
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved validation report to: {args.output}")


if __name__ == "__main__":
    main()
