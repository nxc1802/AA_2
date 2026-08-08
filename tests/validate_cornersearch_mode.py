import os
import sys
import time
import json
import logging
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core import (
    set_seed,
    get_best_device,
    prepare_model_for_eval,
    compute_spatial_l0,
    compute_distortion_metrics
)
from src.datasets.dataset_loader import get_dataloaders
from src.models.model_factory import get_model, find_existing_checkpoint
from src.attacks.adapters.cornersearch_adapter import CornerSearchOfficialAdapter
from src.benchmark.run_attack_benchmark import get_stratified_indices

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("CornerSearchValidation")

def run_cornersearch_validation(num_samples=100, seed=42):
    set_seed(seed)
    device = get_best_device()
    ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if not ckpt:
        raise FileNotFoundError("resnet18_cifar10_best.pth checkpoint required for CornerSearch validation.")

    model = get_model("resnet18", pretrained=False, checkpoint_path=ckpt, device=device)
    model = prepare_model_for_eval(model, device)

    _, _, test_loader = get_dataloaders(batch_size=1024)
    test_ds = test_loader.dataset

    indices = get_stratified_indices(test_ds, num_samples, seed=seed)
    eval_loader = DataLoader(Subset(test_ds, indices), batch_size=num_samples, shuffle=False)

    K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

    logger.info(f"Starting CornerSearch Validation Experiment on {num_samples} samples (Device: {device})...")

    # --- Protocol A: Independent Runs ---
    logger.info("Running Protocol A: Independent CornerSearch runs for each K...")
    protocol_a_results = {}
    total_time_a = 0.0

    for K in K_VALUES:
        attacker = CornerSearchOfficialAdapter(model, k=K, device=device)
        t0 = time.time()
        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                clean_preds = torch.argmax(model(x), dim=1)
            c_mask = (clean_preds == y)
            x_adv = attacker.attack(x, y)
            with torch.no_grad():
                adv_preds = torch.argmax(model(x_adv), dim=1)
            succ_mask = c_mask & (adv_preds != y)
            diff = x_adv - x
            l0_per = compute_spatial_l0(diff)

            clean_correct = c_mask.sum().item()
            adv_succ = succ_mask.sum().item()
            cond_asr = 100.0 * adv_succ / max(1, clean_correct)
            mean_l0 = l0_per[succ_mask].float().mean().item() if adv_succ > 0 else 0.0
            queries = sum(attacker.last_queries) if hasattr(attacker, "last_queries") else 0

            protocol_a_results[K] = {
                "cond_asr": cond_asr,
                "mean_l0": mean_l0,
                "queries": queries,
                "succ_count": adv_succ
            }
        dt = time.time() - t0
        total_time_a += dt
        logger.info(f"Protocol A (K={K}): Cond ASR={cond_asr:.2f}%, Success Avg L0={mean_l0:.2f}, Queries={queries}, Time={dt:.2f}s")

    # --- Protocol B: Progressive Run ---
    logger.info("Running Protocol B: Progressive CornerSearch single-pass (K_max=128)...")
    K_max = max(K_VALUES)
    attacker_prog = CornerSearchOfficialAdapter(model, k=K_max, device=device)
    t0 = time.time()
    protocol_b_results = {}

    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_preds = torch.argmax(model(x), dim=1)
        c_mask = (clean_preds == y)
        x_adv = attacker_prog.attack(x, y)
        with torch.no_grad():
            adv_preds = torch.argmax(model(x_adv), dim=1)
        fooled_mask = c_mask & (adv_preds != y)
        diff = x_adv - x
        l0_tensor = compute_spatial_l0(diff)
        clean_correct = c_mask.sum().item()
        queries_prog = sum(attacker_prog.last_queries) if hasattr(attacker_prog, "last_queries") else 0

        for K in K_VALUES:
            succ_k_mask = fooled_mask & (l0_tensor <= K)
            adv_succ_k = succ_k_mask.sum().item()
            cond_asr_k = 100.0 * adv_succ_k / max(1, clean_correct)
            mean_l0_k = l0_tensor[succ_k_mask].float().mean().item() if adv_succ_k > 0 else 0.0

            protocol_b_results[K] = {
                "cond_asr": cond_asr_k,
                "mean_l0": mean_l0_k,
                "queries": queries_prog,
                "succ_count": adv_succ_k
            }
            logger.info(f"Protocol B (K={K}): Cond ASR={cond_asr_k:.2f}%, Success Avg L0={mean_l0_k:.2f}, Queries={queries_prog}")
    total_time_b = time.time() - t0

    # --- Comparison Metrics ---
    logger.info("\n=== CornerSearch Protocol A (Independent) vs Protocol B (Progressive) Comparison ===")
    comparison_rows = []

    for K in K_VALUES:
        a = protocol_a_results[K]
        b = protocol_b_results[K]
        delta_asr = b["cond_asr"] - a["cond_asr"]
        delta_l0 = b["mean_l0"] - a["mean_l0"]
        delta_queries = b["queries"] - a["queries"]

        row = {
            "K": K,
            "Indep_ASR (%)": round(a["cond_asr"], 2),
            "Prog_ASR (%)": round(b["cond_asr"], 2),
            "Delta_ASR (%)": round(delta_asr, 2),
            "Indep_L0": round(a["mean_l0"], 2),
            "Prog_L0": round(b["mean_l0"], 2),
            "Delta_L0": round(delta_l0, 2),
            "Indep_Queries": a["queries"],
            "Prog_Queries": b["queries"],
            "Delta_Queries": delta_queries
        }
        comparison_rows.append(row)

    import pandas as pd
    df_comp = pd.DataFrame(comparison_rows)
    out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "../result/metrics/cornersearch_validation.csv"))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_comp.to_csv(out_csv, index=False)

    logger.info(f"\n{df_comp.to_string(index=False)}")
    logger.info(f"Total Time Protocol A: {total_time_a:.2f}s | Total Time Protocol B: {total_time_b:.2f}s")
    logger.info(f"Saved comparison results to '{out_csv}'")
    return df_comp

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    run_cornersearch_validation(num_samples=num_samples)
