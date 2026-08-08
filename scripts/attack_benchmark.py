import argparse
import os
import yaml
import json
from aa.utils import set_seed, get_best_device, get_git_reproducibility_info
from aa.data import get_sample_batch
from aa.models import get_model
from aa.attacks import create_attack
from aa.benchmark import evaluate_attack


def main():
    parser = argparse.ArgumentParser(description="Run AA Attack Benchmark")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to experiment config YAML")
    parser.add_argument("--output", type=str, default="result/benchmark_results.json", help="Path to output JSON")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_best_device()

    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    bench_cfg = cfg.get("benchmark", {})

    loader = get_sample_batch(
        dataset_name=ds_cfg.get("name", "cifar10"),
        batch_size=ds_cfg.get("batch_size", 64),
        num_samples=ds_cfg.get("samples", 100)
    )

    model = get_model(
        model_name=model_cfg.get("name", "resnet18"),
        checkpoint_path=model_cfg.get("checkpoint", None),
        device=device
    )

    attack_names = cfg.get("attacks", ["pgd", "ours"])
    k_values = bench_cfg.get("k_values", [16])

    all_results = {
        "metadata": {
            "config": cfg,
            "device": str(device),
            "reproducibility": get_git_reproducibility_info()
        },
        "results": {}
    }

    print(f"=== Starting Benchmark on {device} ({len(loader.dataset)} samples) ===")

    for atk_name in attack_names:
        all_results["results"][atk_name] = {}
        for k in k_values:
            print(f"--> Running attack: {atk_name} (K={k})...")
            try:
                attack = create_attack(atk_name, model=model, k=k)
                res = evaluate_attack(model, attack, loader, device=device)
                all_results["results"][atk_name][f"k_{k}"] = res
                print(f"    ASR: {res['asr']:.2f}%, Clean Acc: {res['clean_accuracy']:.2f}%, Runtime: {res['runtime_seconds']:.2f}s")
            except Exception as e:
                print(f"    ⚠️ Failed running {atk_name} (K={k}): {e}")
                all_results["results"][atk_name][f"k_{k}"] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"=== Benchmark Completed! Results saved to {args.output} ===")


if __name__ == "__main__":
    main()
