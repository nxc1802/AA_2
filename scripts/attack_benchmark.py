import argparse
import os
import yaml
import json
from aa.utils import set_seed, get_best_device, get_git_reproducibility_info
from aa.data import get_sample_batch_indices
from aa.models import get_model
from aa.attacks import create_attack, get_attack_spec
from aa.benchmark import evaluate_attack, derive_minimal_asr_curve


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

    loader, sample_indices, sample_hash = get_sample_batch_indices(
        dataset_name=ds_cfg.get("name", "cifar10"),
        batch_size=ds_cfg.get("batch_size", 64),
        num_samples=ds_cfg.get("samples", 100),
        seed=seed
    )

    model = get_model(
        model_name=model_cfg.get("name", "resnet18"),
        checkpoint_path=model_cfg.get("checkpoint", None),
        device=device
    )

    attack_names = cfg.get("attacks", ["pgd", "ours"])
    k_values = bench_cfg.get("k_values", [1, 2, 4, 8, 16, 32, 64])
    attacks_kwargs = cfg.get("attacks_kwargs", {})

    all_results = {
        "metadata": {
            "config": cfg,
            "device": str(device),
            "sample_indices_hash": sample_hash,
            "model_checkpoint_sha256": getattr(model, "checkpoint_sha256", None),
            "reproducibility": get_git_reproducibility_info()
        },
        "results": {}
    }

    print(f"=== Starting Attack Benchmark on {device} ({len(loader.dataset)} samples) ===")

    for atk_item in attack_names:
        if isinstance(atk_item, dict):
            atk_name = list(atk_item.keys())[0]
            atk_kwargs = atk_item[atk_name]
        else:
            atk_name = str(atk_item)
            atk_kwargs = attacks_kwargs.get(atk_name, {})

        spec = get_attack_spec(atk_name)
        mode = spec.mode
        all_results["results"][atk_name] = {}

        if mode == "dense":
            print(f"--> Running DENSE attack: {atk_name} (single pass)...")
            try:
                attack = create_attack(atk_name, model=model, **atk_kwargs)
                res = evaluate_attack(model, attack, loader, device=device)
                all_results["results"][atk_name]["dense"] = res
                print(f"    [DENSE] ASR: {res['asr']:.2f}%, Clean Acc: {res['clean_accuracy']:.2f}%, Runtime: {res['runtime_seconds']:.2f}s")
            except Exception as e:
                print(f"    ⚠️ Failed running dense attack {atk_name}: {e}")
                all_results["results"][atk_name]["dense"] = {"error": str(e)}

        elif mode == "minimal":
            print(f"--> Running MINIMAL support attack: {atk_name} (single pass & deriving ASR@K)...")
            try:
                attack = create_attack(atk_name, model=model, **atk_kwargs)
                base_res = evaluate_attack(model, attack, loader, device=device)
                derived_curve = derive_minimal_asr_curve(base_res, k_values)
                all_results["results"][atk_name] = derived_curve
                print(f"    [MINIMAL] Median L0: {base_res['metrics']['succ_l0_median']}, Derived ASR@16: {derived_curve.get('k_16', {}).get('asr', 0.0):.2f}%")
            except Exception as e:
                print(f"    ⚠️ Failed running minimal attack {atk_name}: {e}")
                all_results["results"][atk_name]["error"] = str(e)

        else: # "budget"
            print(f"--> Running BUDGET attack: {atk_name} (sweeping K={k_values})...")
            for k in k_values:
                try:
                    attack = create_attack(atk_name, model=model, k=k, **atk_kwargs)
                    res = evaluate_attack(model, attack, loader, device=device)
                    all_results["results"][atk_name][f"k_{k}"] = res
                    print(f"    (K={k}) ASR: {res['asr']:.2f}%, Cond Robust Acc: {res['conditional_robust_accuracy']:.2f}%, Runtime: {res['runtime_seconds']:.2f}s")
                except Exception as e:
                    print(f"    ⚠️ Failed running {atk_name} (K={k}): {e}")
                    all_results["results"][atk_name][f"k_{k}"] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"=== Attack Benchmark Completed! Saved to {args.output} ===")


if __name__ == "__main__":
    main()

