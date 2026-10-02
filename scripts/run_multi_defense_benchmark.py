import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from aa.utils import set_seed, get_best_device
from aa.data import get_sample_batch_indices
from aa.models import get_model
from aa.defenses import (
    GaussianBlurDefense,
    MedianFilterDefense,
    JPEGDefense,
    DefendedModelAdapter
)
from aa.attacks import create_attack
from aa.benchmark import evaluate_attack

DEFENSES = {
    "blur": GaussianBlurDefense(),
    "median": MedianFilterDefense(),
    "jpeg": JPEGDefense(),
}

MODELS = [
    {"name": "resnet18", "ckpt": "result/saved_models/resnet18_cifar10_best.pth"},
    {"name": "mobilenet_v2", "ckpt": "result/saved_models/mobilenet_v2_cifar10_best.pth"},
]

def main():
    set_seed(42)
    device = get_best_device()
    print("=" * 80)
    print(f"DEFENSE GENERALIZATION BENCHMARK (DEF-008) on {device}")
    print("=" * 80)

    loader, sample_indices, sample_hash = get_sample_batch_indices(
        dataset_name="cifar10",
        batch_size=50,
        num_samples=100,
        seed=42
    )

    results = {}

    for m_info in MODELS:
        m_name = m_info["name"]
        ckpt = m_info["ckpt"]
        print(f"\n--> Model Backbone: {m_name.upper()}")

        base_model = get_model(
            model_name=m_name,
            dataset_name="cifar10",
            checkpoint_path=ckpt,
            strict_checkpoint=False,
            device=device,
            eval_mode=True
        )

        results[m_name] = {}

        for def_name, def_layer in DEFENSES.items():
            results[m_name][def_name] = {}
            for mode in ["oblivious", "adaptive"]:
                print(f"    Evaluating [{def_name.upper()}] Defense (Mode: {mode.upper()})...")
                defended_model = DefendedModelAdapter(
                    model=base_model,
                    defense=def_layer,
                    mode=mode
                )

                # Test with CASA @ K=4
                casa_atk = create_attack(
                    "casa", model=defended_model, k=4, steps=10, inner_steps=4,
                    repair_steps=2, alpha=0.25, loss_fn="dlr", drop_and_repair=True,
                    enable_gct=True, spatial_nms=True
                )

                t0 = time.time()
                eval_res = evaluate_attack(defended_model, casa_atk, loader, device=device)
                elapsed = time.time() - t0

                results[m_name][def_name][mode] = {
                    "asr": eval_res["asr"],
                    "clean_acc": eval_res["clean_accuracy"],
                    "robust_acc": eval_res.get("cond_robust_accuracy", 100 - eval_res["asr"]),
                    "runtime_s": round(elapsed, 2)
                }
                print(f"      ASR: {eval_res['asr']:.2f}% | Clean Acc: {eval_res['clean_accuracy']:.2f}% ({elapsed:.1f}s)")

    output_path = "result/multi_defense_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Defense Generalization results to: {output_path}")

if __name__ == "__main__":
    main()
