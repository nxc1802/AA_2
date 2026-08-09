import argparse
import os
import yaml
import json
from aa.utils import set_seed, get_best_device, get_git_reproducibility_info
from aa.data import get_sample_batch_indices
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
    parser.add_argument("--eval-modes", nargs="+", default=["adaptive", "oblivious"], help="Evaluation modes")
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

    loader, sample_indices, sample_hash = get_sample_batch_indices(
        dataset_name=ds_cfg.get("name", "cifar10"),
        batch_size=ds_cfg.get("batch_size", 64),
        num_samples=ds_cfg.get("samples", 100),
        seed=seed
    )

    base_model = get_model(
        model_name=model_cfg.get("name", "resnet18"),
        checkpoint_path=model_cfg.get("checkpoint", None),
        expected_sha256=model_cfg.get("expected_sha256", None),
        device=device
    )

    sparse_attacks = cfg.get("defense_attacks", ["pgd0", "sparse_rs", "ours"])
    k_defense = cfg.get("defense_k", 16)

    all_results = {
        "metadata": {
            "config": cfg,
            "strict_mode": strict_mode,
            "device": str(device),
            "sample_indices_hash": sample_hash,
            "model_checkpoint_sha256": getattr(base_model, "checkpoint_sha256", None),
            "reproducibility": get_git_reproducibility_info()
        },
        "defenses": {}
    }

    print(f"=== Starting Defense Benchmark on {device} ({len(loader.dataset)} samples, strict={strict_mode}) ===")

    for def_name, defense_obj in DEFENSES_MAP.items():
        all_results["defenses"][def_name] = {}
        for mode in args.eval_modes:
            all_results["defenses"][def_name][mode] = {}
            print(f"--> Evaluating Defense: {def_name} (Mode: {mode}, K={k_defense})...")
            defended_model = DefendedModelAdapter(base_model, defense=defense_obj, mode=mode)

            for atk_name in sparse_attacks:
                try:
                    atk_cfg = cfg.get("attacks_kwargs", {}).get(atk_name, {})
                    attack = create_attack(atk_name, model=defended_model, k=k_defense, **atk_cfg)
                    res = evaluate_attack(defended_model, attack, loader, device=device)
                    all_results["defenses"][def_name][mode][atk_name] = res
                    print(f"    [{mode}] {atk_name} -> Defended Clean Acc: {res['clean_accuracy']:.2f}%, Cond Robust Acc: {res['conditional_robust_accuracy']:.2f}%, ASR: {res['asr']:.2f}%")
                except Exception as e:
                    print(f"    ⚠️ Failed {atk_name} on {def_name} [{mode}]: {e}")
                    if strict_mode:
                        raise RuntimeError(f"Attack '{atk_name}' failed on defense '{def_name}' [{mode}] in strict paper mode: {e}") from e
                    all_results["defenses"][def_name][mode][atk_name] = {"error": str(e)}

    # Post-run assertion check for strict mode
    failed_defenses = []
    for def_k, def_v in all_results["defenses"].items():
        for m_k, m_v in def_v.items():
            for atk_k, atk_v in m_v.items():
                if isinstance(atk_v, dict) and "error" in atk_v:
                    failed_defenses.append(f"{def_k}/{m_k}/{atk_k}: {atk_v['error']}")

    if failed_defenses:
        err_msg = f"Defense benchmark finished with {len(failed_defenses)} failed execution(s): {failed_defenses}"
        if strict_mode:
            raise RuntimeError(f"STRICT DEFENSE BENCHMARK FAILED: {err_msg}")
        else:
            print(f"⚠️ WARNING: {err_msg}", flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"=== Defense Benchmark Completed! Saved to {args.output} ===")


if __name__ == "__main__":
    main()

