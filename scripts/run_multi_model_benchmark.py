import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aa.utils import set_seed, get_best_device
from aa.data import get_sample_batch_indices
from aa.models import get_model
from aa.attacks import create_attack
from aa.benchmark import evaluate_attack

MODELS = [
    {"name": "resnet18", "ckpt": "result/saved_models/resnet18_cifar10_best.pth"},
    {"name": "wideresnet28_10", "ckpt": "result/saved_models/wideresnet28_10_cifar10_best.pth"},
    {"name": "mobilenet_v2", "ckpt": "result/saved_models/mobilenet_v2_cifar10_best.pth"},
    {"name": "vit_cifar", "ckpt": "result/saved_models/vit_cifar_cifar10_best.pth"},
]

def main():
    set_seed(42)
    device = get_best_device()
    print("=" * 80)
    print(f"MULTI-ARCHITECTURE BENCHMARK (ATT-041) on {device}")
    print("=" * 80)

    loader, sample_indices, sample_hash = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=50,
        num_samples=100,
        seed=42
    )

    results = {}
    k_values = [1, 2, 4, 8, 16]

    for m_info in MODELS:
        m_name = m_info["name"]
        ckpt = m_info["ckpt"]
        print(f"\n--> Evaluating Architecture: {m_name.upper()}...")

        model = get_model(
            model_name=m_name,
            dataset_name="cifar10",
            checkpoint_path=ckpt,
            strict_checkpoint=False,
            device=device,
            eval_mode=True
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params / 1e6:.2f}M | Checkpoint: {ckpt}")

        results[m_name] = {
            "num_parameters_million": round(n_params / 1e6, 2),
            "attacks": {}
        }

        # 1. Evaluate FGSM & PGD
        for dense_atk in ["fgsm", "pgd"]:
            atk_inst = create_attack(dense_atk, model=model, eps=8/255)
            res = evaluate_attack(model, atk_inst, loader, device=device)
            results[m_name]["attacks"][dense_atk] = {
                "asr": res["asr"],
                "clean_acc": res["clean_accuracy"],
                "runtime_s": res.get("runtime_seconds", 0)
            }
            print(f"    [{dense_atk.upper()}] ASR: {res['asr']:.2f}%, Clean Acc: {res['clean_accuracy']:.2f}%")

        # 2. Evaluate S-PGD vs CASA across K
        for method in ["spgd", "casa"]:
            results[m_name]["attacks"][method] = {}
            for k in k_values:
                if method == "spgd":
                    atk_inst = create_attack("spgd", model=model, k=k, steps=20, alpha=0.25)
                else:
                    atk_inst = create_attack(
                        "casa", model=model, k=k, steps=10, inner_steps=4,
                        repair_steps=2, alpha=0.25, loss_fn="dlr", drop_and_repair=True,
                        enable_gct=True, spatial_nms=True
                    )
                
                t0 = time.time()
                res = evaluate_attack(model, atk_inst, loader, device=device)
                elapsed = time.time() - t0
                
                results[m_name]["attacks"][method][f"k_{k}"] = {
                    "asr": res["asr"],
                    "cond_robust_acc": res.get("cond_robust_accuracy", 100 - res["asr"]),
                    "runtime_s": elapsed
                }
                print(f"    [{method.upper()} @ K={k:2d}] ASR: {res['asr']:5.2f}% | Cond Robust Acc: {res.get('cond_robust_accuracy', 0):5.2f}% ({elapsed:.1f}s)")

    output_path = "result/multi_model_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Multi-Architecture Benchmark results to: {output_path}")

if __name__ == "__main__":
    main()
