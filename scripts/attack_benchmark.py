import argparse
import os
import yaml
import json
from aa.utils import set_seed, get_best_device, get_git_reproducibility_info
from aa.data import get_sample_batch_indices
from aa.models import get_model
from aa.attacks import create_attack, get_attack_spec
from aa.benchmark import evaluate_attack, derive_minimal_asr_curve, derive_progressive_asr_curve


def main():
    parser = argparse.ArgumentParser(description="Run AA Attack Benchmark")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to experiment config YAML")
    parser.add_argument("--output", type=str, default="result/benchmark_results.json", help="Path to output JSON")
    parser.add_argument("--strict", action="store_true", help="Fail-fast if any attack execution fails (recommended for paper runs)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    strict_mode = args.strict or cfg.get("strict", False)
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
        expected_sha256=model_cfg.get("expected_sha256", None),
        device=device
    )

    attack_names = cfg.get("attacks", ["pgd", "ours"])
    k_values = bench_cfg.get("k_values", [1, 2, 4, 8, 16, 32, 64])
    attacks_kwargs = cfg.get("attacks_kwargs", {})

    all_results = {
        "metadata": {
            "config": cfg,
            "strict_mode": strict_mode,
            "device": str(device),
            "sample_indices_hash": sample_hash,
            "model_checkpoint_sha256": getattr(model, "checkpoint_sha256", None),
            "reproducibility": get_git_reproducibility_info()
        },
        "results": {}
    }

    print(f"=== Starting Attack Benchmark on {device} ({len(loader.dataset)} samples, strict={strict_mode}) ===", flush=True)

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
            print(f"--> Running DENSE attack: {atk_name} (single pass)...", flush=True)
            try:
                attack = create_attack(atk_name, model=model, **atk_kwargs)
                res = evaluate_attack(model, attack, loader, device=device)
                all_results["results"][atk_name]["dense"] = res
                print(f"    [DENSE] ASR: {res['asr']:.2f}%, Clean Acc: {res['clean_accuracy']:.2f}%, Runtime: {res['runtime_seconds']:.2f}s", flush=True)
            except Exception as e:
                print(f"    ⚠️ Failed running dense attack {atk_name}: {e}", flush=True)
                if strict_mode:
                    raise RuntimeError(f"Dense attack '{atk_name}' failed in strict paper benchmark mode: {e}") from e
                all_results["results"][atk_name]["dense"] = {"error": str(e)}

        elif mode == "progressive":
            k_max = max(k_values)
            print(f"--> Running PROGRESSIVE attack: {atk_name} (single pass at Kmax={k_max} & deriving curve)...", flush=True)
            try:
                clean_atk_kwargs = {k: v for k, v in atk_kwargs.items() if k not in ("k", "max_k")}
                attack = create_attack(atk_name, model=model, k=k_max, **clean_atk_kwargs)
                base_res = evaluate_attack(model, attack, loader, device=device)
                derived_curve = derive_progressive_asr_curve(base_res, k_values)
                all_results["results"][atk_name] = derived_curve
                k_max_asr = derived_curve.get(f"k_{k_max}", {}).get("asr", 0.0)
                print(f"    [PROGRESSIVE] ASR@{k_max}: {k_max_asr:.2f}%, Clean Acc: {base_res['clean_accuracy']:.2f}%, Runtime: {base_res['runtime_seconds']:.2f}s", flush=True)
            except Exception as e:
                print(f"    ⚠️ Failed running progressive attack {atk_name}: {e}", flush=True)
                if strict_mode:
                    raise RuntimeError(f"Progressive attack '{atk_name}' failed in strict paper benchmark mode: {e}") from e
                all_results["results"][atk_name]["error"] = str(e)

        elif mode == "minimal":
            print(f"--> Running MINIMAL support attack: {atk_name} (single pass & deriving ASR@K)...", flush=True)
            try:
                attack = create_attack(atk_name, model=model, **atk_kwargs)
                base_res = evaluate_attack(model, attack, loader, device=device)
                derived_curve = derive_minimal_asr_curve(base_res, k_values)
                all_results["results"][atk_name] = derived_curve
                print(f"    [MINIMAL] Median L0: {base_res['metrics']['succ_l0_median']}, Derived ASR@16: {derived_curve.get('k_16', {}).get('asr', 0.0):.2f}%", flush=True)
            except Exception as e:
                print(f"    ⚠️ Failed running minimal attack {atk_name}: {e}", flush=True)
                if strict_mode:
                    raise RuntimeError(f"Minimal attack '{atk_name}' failed in strict paper benchmark mode: {e}") from e
                all_results["results"][atk_name]["error"] = str(e)

        else: # "budget"
            print(f"--> Running BUDGET attack: {atk_name} (sweeping K={k_values})...", flush=True)
            for k in k_values:
                try:
                    clean_atk_kwargs = {key: val for key, val in atk_kwargs.items() if key not in ("k", "max_k")}
                    attack = create_attack(atk_name, model=model, k=k, **clean_atk_kwargs)
                    res = evaluate_attack(model, attack, loader, device=device)
                    all_results["results"][atk_name][f"k_{k}"] = res
                    print(f"    (K={k}) ASR: {res['asr']:.2f}%, Cond Robust Acc: {res['conditional_robust_accuracy']:.2f}%, Runtime: {res['runtime_seconds']:.2f}s", flush=True)
                except Exception as e:
                    print(f"    ⚠️ Failed running {atk_name} (K={k}): {e}", flush=True)
                    if strict_mode:
                        raise RuntimeError(f"Budget attack '{atk_name}' (K={k}) failed in strict paper benchmark mode: {e}") from e
                    all_results["results"][atk_name][f"k_{k}"] = {"error": str(e)}

    # Post-run assertion check for strict mode
    failed_attacks = []
    for atk_key, atk_val in all_results["results"].items():
        if isinstance(atk_val, dict):
            if "error" in atk_val:
                failed_attacks.append(f"{atk_key}: {atk_val['error']}")
            else:
                for sub_k, sub_val in atk_val.items():
                    if isinstance(sub_val, dict) and "error" in sub_val:
                        failed_attacks.append(f"{atk_key}/{sub_k}: {sub_val['error']}")

    if failed_attacks:
        err_msg = f"Benchmark finished with {len(failed_attacks)} failed attack execution(s): {failed_attacks}"
        if strict_mode:
            raise RuntimeError(f"STRICT BENCHMARK FAILED: {err_msg}")
        else:
            print(f"⚠️ WARNING: {err_msg}", flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"=== Attack Benchmark Completed! Saved to {args.output} ===")


if __name__ == "__main__":
    main()

