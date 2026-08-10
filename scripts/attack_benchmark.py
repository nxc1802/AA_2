import argparse
import os
import yaml
import json
import torch
from aa.utils import set_seed, get_best_device, enable_gpu_optimizations, get_git_reproducibility_info
from aa.data import get_sample_batch_indices
from aa.models import get_model
from aa.attacks import create_attack, get_attack_spec
from aa.benchmark import evaluate_attack, derive_minimal_asr_curve, derive_progressive_asr_curve
from aa.cache import AttackArtifactCache
from aa.scheduler import MultiGPUScheduler


def main():
    parser = argparse.ArgumentParser(description="Run AA Attack Benchmark")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to experiment config YAML")
    parser.add_argument("--output", type=str, default="result/benchmark_results.json", help="Path to output JSON")
    parser.add_argument("--strict", action="store_true", help="Fail-fast if any attack execution fails (recommended for paper runs)")
    parser.add_argument("--attacks", type=str, default=None, help="Comma-separated list of attack names to run (e.g. 'casa' or 'ours_v2,ours')")
    parser.add_argument("--cache-dir", type=str, default="result/attack_cache", help="Path to attack artifact cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable reading/writing attack artifact cache")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use for multi-GPU sharded execution (default: all available)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Enable CUDA/cuDNN optimizations
    enable_gpu_optimizations()

    strict_mode = args.strict or cfg.get("strict", False)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_best_device()

    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    bench_cfg = cfg.get("benchmark", {})
    attacks_batch_size = cfg.get("attacks_batch_size", {})
    default_batch_size = ds_cfg.get("batch_size", 512)

    cache = None if args.no_cache else AttackArtifactCache(cache_dir=args.cache_dir)
    gpu_scheduler = MultiGPUScheduler(gpus=list(range(args.num_gpus)) if args.num_gpus else None)

    # Default initial loader with default batch size 512
    default_loader, sample_indices, sample_hash = get_sample_batch_indices(
        dataset_name=ds_cfg.get("name", "cifar10"),
        batch_size=default_batch_size,
        num_samples=ds_cfg.get("samples", 1000),
        seed=seed
    )

    model = get_model(
        model_name=model_cfg.get("name", "resnet18"),
        checkpoint_path=model_cfg.get("checkpoint", None),
        expected_sha256=model_cfg.get("expected_sha256", None),
        device=device
    )

    if args.attacks:
        attack_names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    else:
        attack_names = cfg.get("attacks", ["pgd", "ours"])
    k_values = bench_cfg.get("k_values", [1, 2, 4, 8, 16, 32, 64])
    attacks_kwargs = cfg.get("attacks_kwargs", {})

    model_id = getattr(model, "checkpoint_sha256", None) or model_cfg.get("name", "resnet18")

    repro_info = get_git_reproducibility_info()
    git_commit = repro_info.get("git_commit", "unknown")

    all_results = {
        "metadata": {
            "config": cfg,
            "strict_mode": strict_mode,
            "device": str(device),
            "num_gpus": gpu_scheduler.num_gpus,
            "sample_indices_hash": sample_hash,
            "model_checkpoint_sha256": getattr(model, "checkpoint_sha256", None),
            "reproducibility": repro_info
        },
        "results": {}
    }

    print(f"=== Starting Attack Benchmark on {device} ({len(default_loader.dataset)} samples, GPUs={gpu_scheduler.num_gpus}, strict={strict_mode}) ===", flush=True)

    for atk_item in attack_names:
        if isinstance(atk_item, dict):
            atk_name = list(atk_item.keys())[0]
            atk_kwargs = atk_item[atk_name]
        else:
            atk_name = str(atk_item)
            atk_kwargs = attacks_kwargs.get(atk_name, {})

        # Lookup per-method batch size from config (default: 512)
        atk_batch_size = attacks_batch_size.get(atk_name, default_batch_size)
        if atk_batch_size != default_batch_size:
            loader, _, _ = get_sample_batch_indices(
                dataset_name=ds_cfg.get("name", "cifar10"),
                batch_size=atk_batch_size,
                num_samples=ds_cfg.get("samples", 1000),
                seed=seed
            )
        else:
            loader = default_loader

        spec = get_attack_spec(atk_name)
        mode = spec.mode
        all_results["results"][atk_name] = {}

        def _run_single_or_multigpu(target_attack_name: str, target_k: Optional[int], target_kwargs: dict) -> Dict[str, Any]:
            cache_key = cache.compute_cache_key(
                dataset_hash=sample_hash,
                model_identifier=model_id,
                attack_name=target_attack_name,
                attack_kwargs=target_kwargs,
                seed=seed,
                k=target_k,
                git_commit=git_commit
            ) if cache else None

            if gpu_scheduler.is_multi_gpu() and (cache is None or not cache.has(cache_key)):
                # Multi-GPU execution: shard evaluation across available GPUs
                def model_fn():
                    return get_model(
                        model_name=model_cfg.get("name", "resnet18"),
                        checkpoint_path=model_cfg.get("checkpoint", None),
                        expected_sha256=model_cfg.get("expected_sha256", None)
                    )

                def attack_factory(m):
                    kwargs_to_pass = dict(target_kwargs)
                    if target_k is not None:
                        kwargs_to_pass["k"] = target_k
                    return create_attack(target_attack_name, model=m, **kwargs_to_pass)

                sharded_output = gpu_scheduler.run_sharded_attack(
                    model_fn=model_fn,
                    attack_factory=attack_factory,
                    dataset=loader.dataset,
                    batch_size=atk_batch_size
                )
                if cache and cache_key:
                    cache.put(cache_key, sharded_output)

            kwargs_to_pass = dict(target_kwargs)
            if target_k is not None:
                kwargs_to_pass["k"] = target_k
            attack_inst = create_attack(target_attack_name, model=model, **kwargs_to_pass)
            return evaluate_attack(model, attack_inst, loader, device=device, cache=cache, cache_key=cache_key)

        if mode == "dense":
            print(f"--> Running DENSE attack: {atk_name} (batch_size={atk_batch_size})...", flush=True)
            try:
                clean_atk_kwargs = dict(atk_kwargs)
                res = _run_single_or_multigpu(atk_name, None, clean_atk_kwargs)
                all_results["results"][atk_name]["dense"] = res
                print(f"    [DENSE] ASR: {res['asr']:.2f}%, Clean Acc: {res['clean_accuracy']:.2f}%, Gen Runtime: {res['attack_generation_runtime']:.2f}s, Cache Hit: {res['cache_hit']}", flush=True)
            except Exception as e:
                print(f"    ⚠️ Failed running dense attack {atk_name}: {e}", flush=True)
                if strict_mode:
                    raise RuntimeError(f"Dense attack '{atk_name}' failed in strict paper benchmark mode: {e}") from e
                all_results["results"][atk_name]["dense"] = {"error": str(e)}

        elif mode == "progressive":
            k_max = max(k_values)
            print(f"--> Running PROGRESSIVE attack: {atk_name} (Kmax={k_max}, batch_size={atk_batch_size})...", flush=True)
            try:
                clean_atk_kwargs = {k: v for k, v in atk_kwargs.items() if k not in ("k", "max_k")}
                base_res = _run_single_or_multigpu(atk_name, k_max, clean_atk_kwargs)
                derived_curve = derive_progressive_asr_curve(base_res, k_values)
                all_results["results"][atk_name] = derived_curve
                k_max_asr = derived_curve.get(f"k_{k_max}", {}).get("asr", 0.0)
                print(f"    [PROGRESSIVE] ASR@{k_max}: {k_max_asr:.2f}%, Clean Acc: {base_res['clean_accuracy']:.2f}%, Gen Runtime: {base_res['attack_generation_runtime']:.2f}s, Cache Hit: {base_res['cache_hit']}", flush=True)
            except Exception as e:
                print(f"    ⚠️ Failed running progressive attack {atk_name}: {e}", flush=True)
                if strict_mode:
                    raise RuntimeError(f"Progressive attack '{atk_name}' failed in strict paper benchmark mode: {e}") from e
                all_results["results"][atk_name]["error"] = str(e)

        elif mode == "minimal":
            print(f"--> Running MINIMAL support attack: {atk_name} (batch_size={atk_batch_size})...", flush=True)
            try:
                clean_atk_kwargs = dict(atk_kwargs)
                base_res = _run_single_or_multigpu(atk_name, None, clean_atk_kwargs)
                derived_curve = derive_minimal_asr_curve(base_res, k_values)
                all_results["results"][atk_name] = derived_curve
                print(f"    [MINIMAL] Median L0: {base_res['metrics']['succ_l0_median']}, Derived ASR@16: {derived_curve.get('k_16', {}).get('asr', 0.0):.2f}%, Cache Hit: {base_res['cache_hit']}", flush=True)
            except Exception as e:
                print(f"    ⚠️ Failed running minimal attack {atk_name}: {e}", flush=True)
                if strict_mode:
                    raise RuntimeError(f"Minimal attack '{atk_name}' failed in strict paper benchmark mode: {e}") from e
                all_results["results"][atk_name]["error"] = str(e)

        else: # "budget"
            print(f"--> Running BUDGET attack: {atk_name} (sweeping K={k_values}, batch_size={atk_batch_size})...", flush=True)
            for k in k_values:
                try:
                    clean_atk_kwargs = {key: val for key, val in atk_kwargs.items() if key not in ("k", "max_k")}
                    res = _run_single_or_multigpu(atk_name, k, clean_atk_kwargs)
                    all_results["results"][atk_name][f"k_{k}"] = res
                    print(f"    (K={k}) ASR: {res['asr']:.2f}%, Cond Robust Acc: {res['conditional_robust_accuracy']:.2f}%, Gen Runtime: {res['attack_generation_runtime']:.2f}s, Cache Hit: {res['cache_hit']}", flush=True)
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

