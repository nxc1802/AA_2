import argparse
import os
import sys
import time
import json
import torch
from typing import List, Dict, Any

from aa.utils import set_seed, get_best_device, enable_gpu_optimizations
from aa.data import get_sample_batch_indices
from aa.models import get_model
from aa.attacks import create_attack
from aa.benchmark import evaluate_attack


def parse_args():
    parser = argparse.ArgumentParser(description="Run CASA Benchmark on CIFAR-10 with ResNet-18")
    parser.add_argument("--samples", type=int, default=1000, help="Number of CIFAR-10 samples (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name (default: resnet18)")
    parser.add_argument("--checkpoint", type=str, default="result/saved_models/resnet18_cifar10_best.pth")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64], help="K budget values")
    parser.add_argument("--steps", type=int, default=20, help="Outer support exchange steps")
    parser.add_argument("--inner-steps", type=int, default=10, help="Inner RGB optimization steps")
    parser.add_argument("--repair-steps", type=int, default=4, help="Repair steps during swap")
    parser.add_argument("--alpha", type=float, default=0.25, help="Inner step size")
    parser.add_argument("--loss-fn", type=str, default="dlr", choices=["dlr", "margin", "ce"], help="Loss function")
    parser.add_argument("--no-drop-repair", action="store_true", help="Disable Drop-and-Repair minimization")
    parser.add_argument("--no-gct", action="store_true", help="Disable Gradient-Guided Corner Traversal")
    parser.add_argument("--no-spatial-nms", action="store_true", help="Disable Spatial NMS coalition initialization")
    parser.add_argument("--no-adaptive-batch-swap", action="store_true", help="Disable Adaptive Batch Swap fallback")
    parser.add_argument("--pair-search-every", type=int, default=3, help="Frequency of pair exploration steps")
    parser.add_argument("--output", type=str, default="result/casa_benchmark_1000_bs16.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    enable_gpu_optimizations()
    set_seed(args.seed)
    device = get_best_device()

    print("=" * 80)
    print(f"CASA BENCHMARK | Dataset: CIFAR-10 ({args.samples} samples, BS={args.batch_size})")
    print(f"Model: {args.model} | Device: {device} | K values: {args.k_values}")
    print(f"CASA Settings: steps={args.steps}, inner_steps={args.inner_steps}, repair_steps={args.repair_steps}, alpha={args.alpha}, loss_fn={args.loss_fn}, drop_and_repair={not args.no_drop_repair}")
    print("=" * 80, flush=True)

    # 1. Load Data
    loader, sample_indices, sample_hash = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=args.batch_size,
        num_samples=args.samples,
        seed=args.seed
    )

    # 2. Load Model
    model = get_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        strict_checkpoint=False,
        device=device,
        eval_mode=True
    )

    results = {}
    total_start = time.time()

    for k in args.k_values:
        print(f"\n--> [Running K={k:2d}] ...", flush=True)
        t0 = time.time()

        attack_kwargs = dict(
            k=k,
            steps=args.steps,
            inner_steps=args.inner_steps,
            repair_steps=args.repair_steps,
            alpha=args.alpha,
            loss_fn=args.loss_fn,
            drop_and_repair=not args.no_drop_repair,
            enable_gct=not args.no_gct,
            spatial_nms=not args.no_spatial_nms,
            pair_search_every=args.pair_search_every,
            adaptive_batch_swap=not args.no_adaptive_batch_swap,
        )

        attack_inst = create_attack("casa", model=model, strict=False, **attack_kwargs)
        eval_res = evaluate_attack(model, attack_inst, loader, device=device)
        elapsed = time.time() - t0

        asr = eval_res["asr"]
        cra = eval_res["conditional_robust_accuracy"]
        clean_acc = eval_res["clean_accuracy"]
        metrics = eval_res.get("metrics", {})
        mean_l0 = metrics.get("succ_l0_mean", 0.0)
        med_l0 = metrics.get("succ_l0_median", 0.0)
        mean_l2 = metrics.get("succ_l2_mean", 0.0)
        mean_linf = metrics.get("succ_linf_mean", 0.0)
        mean_psnr = metrics.get("succ_psnr_mean", 0.0)
        mean_ssim = metrics.get("succ_ssim_mean", 0.0)

        results[f"k_{k}"] = eval_res

        print(f"    ✓ K={k:2d} | ASR: {asr:6.2f}% | Cond Robust Acc: {cra:6.2f}% | Mean L0: {mean_l0:5.2f} (Med: {med_l0}) | Linf: {mean_linf:.4f} | Time: {elapsed:5.1f}s", flush=True)

    total_time = time.time() - total_start

    # Summary Table
    print("\n" + "=" * 85)
    print("FINAL CASA BENCHMARK SUMMARY (1000 Samples, BS=16, ResNet-18)")
    print("=" * 85)
    header = f"| {'K':^4} | {'ASR (%)':^9} | {'CRA (%)':^9} | {'Mean L0':^9} | {'Median L0':^9} | {'Mean L2':^9} | {'Mean Linf':^9} | {'Time (s)':^8} |"
    sep = f"|{'-'*6}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*10}|"
    print(header)
    print(sep)

    for k in args.k_values:
        res = results[f"k_{k}"]
        m = res.get("metrics", {})
        print(f"| {k:^4} | {res['asr']:^9.2f} | {res['conditional_robust_accuracy']:^9.2f} | {m.get('succ_l0_mean', 0.0):^9.2f} | {m.get('succ_l0_median', 0.0):^9.1f} | {m.get('succ_l2_mean', 0.0):^9.4f} | {m.get('succ_linf_mean', 0.0):^9.4f} | {res['attack_generation_runtime']:^8.1f} |")

    print(f"\nTotal Execution Time: {total_time:.1f}s")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "metadata": {
                "samples": args.samples,
                "batch_size": args.batch_size,
                "model": args.model,
                "device": str(device),
                "settings": {
                    "steps": args.steps,
                    "inner_steps": args.inner_steps,
                    "repair_steps": args.repair_steps,
                    "alpha": args.alpha,
                    "loss_fn": args.loss_fn,
                    "drop_and_repair": not args.no_drop_repair
                },
                "total_time": total_time
            },
            "results": results
        }, f, indent=2)

    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
