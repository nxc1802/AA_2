import argparse
import os
import yaml
import json
from aa.utils import set_seed, get_best_device, get_git_reproducibility_info
from aa.data import get_sample_batch
from aa.models import get_model
from aa.defenses import (
    GaussianBlurDefense,
    MedianFilterDefense,
    JPEGDefense,
    TVMDefense,
    DefendedModelAdapter
)
from aa.attacks import create_attack
from aa.benchmark import evaluate_attack


DEFENSES_MAP = {
    "blur": GaussianBlurDefense(),
    "median": MedianFilterDefense(),
    "jpeg": JPEGDefense(),
    "tvm": TVMDefense(),
}


def main():
    parser = argparse.ArgumentParser(description="Run AA Defense Benchmark")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="result/defense_benchmark_results.json", help="Path to output JSON")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = get_best_device()

    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    loader = get_sample_batch(
        dataset_name=ds_cfg.get("name", "cifar10"),
        batch_size=ds_cfg.get("batch_size", 64),
        num_samples=ds_cfg.get("samples", 100)
    )

    base_model = get_model(
        model_name=model_cfg.get("name", "resnet18"),
        checkpoint_path=model_cfg.get("checkpoint", None),
        device=device
    )

    all_results = {
        "metadata": {
            "config": cfg,
            "device": str(device),
            "reproducibility": get_git_reproducibility_info()
        },
        "defenses": {}
    }

    print(f"=== Starting Defense Benchmark on {device} ===")

    for def_name, defense_obj in DEFENSES_MAP.items():
        print(f"--> Evaluating Defense: {def_name}...")
        defended_model = DefendedModelAdapter(base_model, defense=defense_obj, mode="adaptive")

        atk = create_attack("pgd", model=defended_model, k=16)
        res = evaluate_attack(defended_model, atk, loader, device=device)

        all_results["defenses"][def_name] = res
        print(f"    Defended Clean Acc: {res['clean_accuracy']:.2f}%, Robust Acc: {res['robust_accuracy']:.2f}%")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"=== Defense Benchmark Completed! Saved to {args.output} ===")


if __name__ == "__main__":
    main()
