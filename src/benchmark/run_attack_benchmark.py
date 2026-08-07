# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import time
import math
import json
import logging
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd

try:
    import lpips
except ImportError:
    lpips = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core import (
    compute_spatial_l0,
    compute_per_sample_psnr,
    compute_per_sample_ssim,
    compute_per_sample_lpips,
    compute_distortion_metrics,
    prepare_model_for_eval,
    set_seed
)
from src.datasets.dataset_loader import get_dataloaders
from src.models.model_factory import get_model, find_existing_checkpoint

# Import baseline & non-pixel-K attacks (Group C)
from src.attacks.baselines.fgsm import FGSMAttack
from src.attacks.baselines.bim import BIMAttack
from src.attacks.baselines.pgd import PGDAttack
from src.attacks.frequency.sfa import SpectralFrequencyAttack

# Import custom PyTorch implementations
from src.attacks.classical.jsma import JSMAAttack
from src.attacks.classical.onepixel import OnePixelAttack
from src.attacks.classical.corner_search import CornerSearchAttack
from src.attacks.optimization.saif import SAIFAttack
from src.attacks.optimization.pgd0 import PGD0Attack
from src.attacks.optimization.sparse_pgd import SparsePGDAttack
from src.attacks.attention_attribution.ipfsa import IPFSAttack
from src.attacks.attention_attribution.gradient_guidance import GradientGuidanceAttack
from src.attacks.blackbox.sparse_rs import SparseRSAttack
from src.attacks.blackbox.brusle import BruSLeAttack
from src.attacks.proposed.cpa import CooperativePixelsAttack
from src.attacks.proposed.fcsa import FunctionalCoalitionSparseAttack
from src.attacks.proposed.fmsa import FeatureToMinimalSupportAttack
from src.attacks.proposed.hsa import HypergraphSparseAttack

# Import unconstrained minimal support attacks (Group B)
from src.attacks.classical.sparsefool import SparseFoolAttack
from src.attacks.optimization.sigma_zero import SigmaZeroAttack
from src.attacks.optimization.homotopy import HomotopyAttack
from src.attacks.optimization.gse import GroupSparseAttack
from src.attacks.blackbox.pixle import PixleAttack

# Import Official Author Adapters (third_party)
try:
    from src.attacks.adapters import (
        SparseRSOfficialAdapter,
        CornerSearchOfficialAdapter,
        PGD0OfficialAdapter,
        SparseFoolOfficialAdapter,
        SigmaZeroOfficialAdapter,
        SparsePGDOfficialAdapter,
        HomotopyOfficialAdapter,
        GSEOfficialAdapter
    )
    OFFICIAL_ADAPTERS_AVAILABLE = True
except Exception as e:
    OFFICIAL_ADAPTERS_AVAILABLE = False

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS
# ==============================================================================
NUM_BENCHMARK_TEST_SAMPLES = 1000
EVAL_BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../result"))
METRICS_DIR = os.path.join(RESULT_DIR, "metrics")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("AttackBenchmark")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "attack_benchmark.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# EXPERIMENTAL BENCHMARK ENGINE (GROUPS A, B, C)
# ==============================================================================
def run_attack_benchmark_suite(
    model, 
    test_loader, 
    eval_batch_size=EVAL_BATCH_SIZE, 
    num_samples=NUM_BENCHMARK_TEST_SAMPLES, 
    device=DEVICE, 
    seed=42, 
    use_official_adapters=False,
    output_prefix="benchmark"
):
    set_seed(seed)
    model = prepare_model_for_eval(model, device)

    test_ds = test_loader.dataset
    total_ds_len = len(test_ds)
    n_eval = min(num_samples, total_ds_len) if num_samples else total_ds_len

    # Fixed Stratified / Deterministic Sample Indexing
    indices_file = os.path.join(RESULT_DIR, f"benchmark_indices_seed{seed}.json")
    if os.path.exists(indices_file):
        with open(indices_file, "r") as f:
            sampled_indices = json.load(f)[:n_eval]
    else:
        g = torch.Generator().manual_seed(seed)
        sampled_indices = torch.randperm(total_ds_len, generator=g).tolist()[:n_eval]
        with open(indices_file, "w") as f:
            json.dump(sampled_indices, f)

    test_subset = Subset(test_ds, sampled_indices)
    eval_loader = DataLoader(test_subset, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    logger.info(f"Evaluation DataLoader created with batch_size={eval_batch_size}, total_samples={len(test_subset)} (Indices hash: {hashlib.md5(str(sampled_indices).encode()).hexdigest()[:8]})")

    lpips_fn = None
    if lpips is not None:
        try:
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            prepare_model_for_eval(lpips_fn, device)
        except Exception as e:
            logger.warning(f"Failed to initialize LPIPS metric: {e}")
            lpips_fn = None

    K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
    results_list = []
    full_csv_path = os.path.join(METRICS_DIR, f"{output_prefix}_full_attack_metrics.csv")

    # Save benchmark run metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "num_samples": n_eval,
        "use_official_adapters": use_official_adapters,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "sample_indices_md5": hashlib.md5(str(sampled_indices).encode()).hexdigest(),
    }
    with open(os.path.join(METRICS_DIR, f"{output_prefix}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # ==========================================================================
    # GROUP C: Non-pixel-K Attacks (Dense + Spectral Frequency Domain)
    # ==========================================================================
    logger.info("=== GROUP C: Non-pixel-K Attacks (FGSM, BIM, PGD, SFA) ===")
    group_c_attacks = {
        "FGSM": (FGSMAttack(model, device=device), "custom-reimplementation"),
        "BIM": (BIMAttack(model, device=device), "custom-reimplementation"),
        "PGD": (PGDAttack(model, device=device), "custom-reimplementation"),
        "SFA": (SpectralFrequencyAttack(model, freq_k=15, device=device), "custom-reimplementation")
    }

    for name, (attacker, source_label) in group_c_attacks.items():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        
        all_l0s, all_l2s, all_linfs = [], [], []
        all_psnrs, all_ssims, all_lpipss = [], [], []
        all_clean_masks, all_succ_masks = [], []
        total_steps = 0.0

        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            with torch.no_grad():
                clean_preds = torch.argmax(model(x), dim=1)
            c_mask = (clean_preds == y)

            x_adv = attacker.attack(x, y)
            
            if hasattr(attacker, "last_steps"):
                total_steps += sum(attacker.last_steps)
            else:
                steps = getattr(attacker, "steps", getattr(attacker, "max_iter", 1))
                total_steps += steps * B

            with torch.no_grad():
                adv_preds = torch.argmax(model(x_adv), dim=1)

            r_mask = (adv_preds == y)
            succ_mask = c_mask & (~r_mask)

            diff = x_adv - x
            l0_per = compute_spatial_l0(diff)
            l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
            linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

            psnr_per = compute_per_sample_psnr(x, x_adv)
            ssim_per = compute_per_sample_ssim(x, x_adv)
            lpips_per = compute_per_sample_lpips(x, x_adv, lpips_fn)

            all_clean_masks.append(c_mask)
            all_succ_masks.append(succ_mask)
            all_l0s.append(l0_per)
            all_l2s.append(l2_per)
            all_linfs.append(linf_per)
            all_psnrs.append(psnr_per)
            all_ssims.append(ssim_per)
            if lpips_per is not None:
                all_lpipss.append(lpips_per)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        c_mask_tensor = torch.cat(all_clean_masks)
        succ_mask_tensor = torch.cat(all_succ_masks)
        l0_tensor = torch.cat(all_l0s)
        l2_tensor = torch.cat(all_l2s)
        linf_tensor = torch.cat(all_linfs)
        psnr_tensor = torch.cat(all_psnrs)
        ssim_tensor = torch.cat(all_ssims)
        lpips_tensor = torch.cat(all_lpipss) if all_lpipss else None

        clean_correct = c_mask_tensor.sum().item()
        adv_succ = succ_mask_tensor.sum().item()
        total_count = l0_tensor.numel()

        clean_acc = 100.0 * clean_correct / total_count
        cond_asr = 100.0 * adv_succ / max(1, clean_correct)
        rob_acc = 100.0 * (clean_correct - adv_succ) / total_count

        dist_m = compute_distortion_metrics(
            l0_tensor, l2_tensor, linf_tensor, psnr_tensor, ssim_tensor, lpips_tensor, succ_mask_tensor
        )

        res = {
            "Group": "Group C", "Attack Method": name, "K": "N/A",
            "Implementation Source": source_label,
            "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc, 2),
            "Conditional ASR (%)": round(cond_asr, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc, 2),
            "All Avg L0": round(dist_m["all_l0_mean"], 2),
            "Success Avg L0": round(dist_m["succ_l0_mean"], 2),
            "Success Median L0": round(dist_m["succ_l0_median"], 2),
            "Success Avg L2": round(dist_m["succ_l2_mean"], 4),
            "Success Avg L_inf": round(dist_m["succ_linf_mean"], 4),
            "Success PSNR (dB)": round(dist_m["succ_psnr_mean"], 2),
            "Success SSIM": round(dist_m["succ_ssim_mean"], 4),
            "Success LPIPS": round(dist_m["succ_lpips_mean"], 4) if dist_m["succ_lpips_mean"] is not None else None,
            "Avg Iterations": round(total_steps / total_count, 2),
            "Time/Img (s)": round(dt / total_count, 4)
        }
        logger.info(f"[Group C] {name}: Conditional ASR={res['Conditional ASR (%)']}%, Robust Acc={res['Robust Acc (%)']}%, Success Avg L0={res['Success Avg L0']}")
        results_list.append(res)
        pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    # ==========================================================================
    # GROUP A: Direct K-Sweep Attacks (Budget Constrained)
    # ==========================================================================
    logger.info("=== GROUP A: Direct K-Sweep Attacks (Explicit K-budget) ===")
    if use_official_adapters and OFFICIAL_ADAPTERS_AVAILABLE:
        logger.info(">>> Running Group A with Official Author Adapters (third_party)")
        group_a_factories = {
            "JSMA": (lambda m, k, d: JSMAAttack(m, k=k, device=d), "custom-reimplementation"),
            "OnePixel": (lambda m, k, d: OnePixelAttack(m, k=k, device=d), "custom-reimplementation"),
            "CornerSearch": (lambda m, k, d: CornerSearchOfficialAdapter(m, k=k, device=d), "official-adapter"),
            "SAIF": (lambda m, k, d: SAIFAttack(m, k=k, device=d), "custom-reimplementation"),
            "PGD0": (lambda m, k, d: PGD0OfficialAdapter(m, k=k, device=d), "official-adapter"),
            "Sparse-PGD": (lambda m, k, d: SparsePGDOfficialAdapter(m, sparsity_budget=k, device=d), "official-adapter"),
            "Sparse-RS": (lambda m, k, d: SparseRSOfficialAdapter(m, n_pixels=k, device=d), "official-adapter"),
            "BruSLe": (lambda m, k, d: BruSLeAttack(m, k=k, device=d), "custom-reimplementation"),
            "IPFSA": (lambda m, k, d: IPFSAttack(m, k_pixels=k, device=d), "custom-reimplementation"),
            "GradientGuidance": (lambda m, k, d: GradientGuidanceAttack(m, sparsity_budget=k, device=d), "custom-reimplementation"),
            "CPA": (lambda m, k, d: CooperativePixelsAttack(m, coalition_size=k, device=d), "ours"),
            "FCSA": (lambda m, k, d: FunctionalCoalitionSparseAttack(m, max_coalition_size=k, device=d), "ours"),
            "FMSA-budgeted": (lambda m, k, d: FeatureToMinimalSupportAttack(m, support_budget=k, device=d), "ours"),
            "HSA-budgeted": (lambda m, k, d: HypergraphSparseAttack(m, budget=k, device=d), "ours")
        }
    else:
        group_a_factories = {
            "JSMA": (lambda m, k, d: JSMAAttack(m, k=k, device=d), "custom-reimplementation"),
            "OnePixel": (lambda m, k, d: OnePixelAttack(m, k=k, device=d), "custom-reimplementation"),
            "CornerSearch": (lambda m, k, d: CornerSearchAttack(m, k=k, device=d), "custom-reimplementation"),
            "SAIF": (lambda m, k, d: SAIFAttack(m, k=k, device=d), "custom-reimplementation"),
            "PGD0": (lambda m, k, d: PGD0Attack(m, k=k, device=d), "custom-reimplementation"),
            "Sparse-PGD": (lambda m, k, d: SparsePGDAttack(m, sparsity_budget=k, device=d), "custom-reimplementation"),
            "Sparse-RS": (lambda m, k, d: SparseRSAttack(m, n_pixels=k, device=d), "custom-reimplementation"),
            "BruSLe": (lambda m, k, d: BruSLeAttack(m, k=k, device=d), "custom-reimplementation"),
            "IPFSA": (lambda m, k, d: IPFSAttack(m, k_pixels=k, device=d), "custom-reimplementation"),
            "GradientGuidance": (lambda m, k, d: GradientGuidanceAttack(m, sparsity_budget=k, device=d), "custom-reimplementation"),
            "CPA": (lambda m, k, d: CooperativePixelsAttack(m, coalition_size=k, device=d), "ours"),
            "FCSA": (lambda m, k, d: FunctionalCoalitionSparseAttack(m, max_coalition_size=k, device=d), "ours"),
            "FMSA-budgeted": (lambda m, k, d: FeatureToMinimalSupportAttack(m, support_budget=k, device=d), "ours"),
            "HSA-budgeted": (lambda m, k, d: HypergraphSparseAttack(m, budget=k, device=d), "ours")
        }

    for name, (factory, source_label) in group_a_factories.items():
        for K in K_VALUES:
            attacker = factory(model, K, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            all_l0s, all_l2s, all_linfs = [], [], []
            all_psnrs, all_ssims, all_lpipss = [], [], []
            all_clean_masks, all_succ_masks = [], []
            total_steps = 0.0

            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                B = x.size(0)
                with torch.no_grad():
                    clean_preds = torch.argmax(model(x), dim=1)
                c_mask = (clean_preds == y)

                x_adv = attacker.attack(x, y)
                
                if hasattr(attacker, "last_steps"):
                    total_steps += sum(attacker.last_steps)
                else:
                    steps = getattr(attacker, "steps", getattr(attacker, "max_iter", 1))
                    total_steps += steps * B

                with torch.no_grad():
                    adv_preds = torch.argmax(model(x_adv), dim=1)

                r_mask = (adv_preds == y)
                succ_mask = c_mask & (~r_mask)

                diff = x_adv - x
                l0_per = compute_spatial_l0(diff)
                l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
                linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

                psnr_per = compute_per_sample_psnr(x, x_adv)
                ssim_per = compute_per_sample_ssim(x, x_adv)
                lpips_per = compute_per_sample_lpips(x, x_adv, lpips_fn)

                all_clean_masks.append(c_mask)
                all_succ_masks.append(succ_mask)
                all_l0s.append(l0_per)
                all_l2s.append(l2_per)
                all_linfs.append(linf_per)
                all_psnrs.append(psnr_per)
                all_ssims.append(ssim_per)
                if lpips_per is not None:
                    all_lpipss.append(lpips_per)

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

            c_mask_tensor = torch.cat(all_clean_masks)
            succ_mask_tensor = torch.cat(all_succ_masks)
            l0_tensor = torch.cat(all_l0s)
            l2_tensor = torch.cat(all_l2s)
            linf_tensor = torch.cat(all_linfs)
            psnr_tensor = torch.cat(all_psnrs)
            ssim_tensor = torch.cat(all_ssims)
            lpips_tensor = torch.cat(all_lpipss) if all_lpipss else None

            clean_correct = c_mask_tensor.sum().item()
            adv_succ = succ_mask_tensor.sum().item()
            total_count = l0_tensor.numel()

            clean_acc = 100.0 * clean_correct / total_count
            cond_asr = 100.0 * adv_succ / max(1, clean_correct)
            rob_acc = 100.0 * (clean_correct - adv_succ) / total_count

            dist_m = compute_distortion_metrics(
                l0_tensor, l2_tensor, linf_tensor, psnr_tensor, ssim_tensor, lpips_tensor, succ_mask_tensor
            )

            res = {
                "Group": "Group A", "Attack Method": name, "K": K,
                "Implementation Source": source_label,
                "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc, 2),
                "Conditional ASR (%)": round(cond_asr, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc, 2),
                "All Avg L0": round(dist_m["all_l0_mean"], 2),
                "Success Avg L0": round(dist_m["succ_l0_mean"], 2),
                "Success Median L0": round(dist_m["succ_l0_median"], 2),
                "Success Avg L2": round(dist_m["succ_l2_mean"], 4),
                "Success Avg L_inf": round(dist_m["succ_linf_mean"], 4),
                "Success PSNR (dB)": round(dist_m["succ_psnr_mean"], 2),
                "Success SSIM": round(dist_m["succ_ssim_mean"], 4),
                "Success LPIPS": round(dist_m["succ_lpips_mean"], 4) if dist_m["succ_lpips_mean"] is not None else None,
                "Avg Iterations": round(total_steps / total_count, 2),
                "Time/Img (s)": round(dt / total_count, 4)
            }
            logger.info(f"[Group A] {name} (K={K}): Conditional ASR={res['Conditional ASR (%)']}%, Robust Acc={res['Robust Acc (%)']}%, Success Avg L0={res['Success Avg L0']}")
            results_list.append(res)
            pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    # ==========================================================================
    # GROUP B: Unconstrained Minimum Support Optimization -> Cumulative ASR@K Evaluation
    # ==========================================================================
    logger.info("=== GROUP B: Minimal Support Optimization (Corrected Cumulative ASR@K Evaluation) ===")
    if use_official_adapters and OFFICIAL_ADAPTERS_AVAILABLE:
        logger.info(">>> Running Group B with Official Author Adapters (third_party)")
        group_b_attacks = {
            "SparseFool": (SparseFoolOfficialAdapter(model, k=250, steps=50, device=device), "official-adapter"),
            "SigmaZero": (SigmaZeroOfficialAdapter(model, steps=50, device=device), "official-adapter"),
            "Homotopy": (HomotopyOfficialAdapter(model, target_sparsity=250, steps=50, device=device), "official-adapter"),
            "GSE": (GSEOfficialAdapter(model, group_size=4, max_groups=64, steps=50, device=device), "official-adapter"),
            "Pixle": (PixleAttack(model, n_swaps=20, max_trials=50, device=device), "custom-reimplementation"),
            "FMSA-minimal-support": (FeatureToMinimalSupportAttack(model, support_budget=250, device=device), "ours")
        }
    else:
        group_b_attacks = {
            "SparseFool": (SparseFoolAttack(model, k=250, steps=50, device=device), "custom-reimplementation"),
            "SigmaZero": (SigmaZeroAttack(model, steps=50, device=device), "custom-reimplementation"),
            "Homotopy": (HomotopyAttack(model, target_sparsity=250, steps=50, device=device), "custom-reimplementation"),
            "GSE": (GroupSparseAttack(model, group_size=4, max_groups=64, steps=50, device=device), "custom-reimplementation"),
            "Pixle": (PixleAttack(model, n_swaps=20, max_trials=50, device=device), "custom-reimplementation"),
            "FMSA-minimal-support": (FeatureToMinimalSupportAttack(model, support_budget=250, device=device), "ours")
        }

    for name, (attacker, source_label) in group_b_attacks.items():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        
        all_l0s, all_l2s, all_linfs = [], [], []
        all_psnrs, all_ssims, all_lpipss = [], [], []
        all_clean_masks, all_fooled_masks = [], []
        total_steps = 0.0

        for x, y in eval_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            with torch.no_grad():
                clean_preds = torch.argmax(model(x), dim=1)
            c_mask = (clean_preds == y)

            x_adv = attacker.attack(x, y)
            
            if hasattr(attacker, "last_steps"):
                total_steps += sum(attacker.last_steps)
            else:
                steps = getattr(attacker, "steps", getattr(attacker, "max_iter", 1))
                total_steps += steps * B

            with torch.no_grad():
                adv_preds = torch.argmax(model(x_adv), dim=1)

            fooled_mask = c_mask & (adv_preds != y)
            diff = x_adv - x
            l0_per = compute_spatial_l0(diff)
            l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
            linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

            psnr_per = compute_per_sample_psnr(x, x_adv)
            ssim_per = compute_per_sample_ssim(x, x_adv)
            lpips_per = compute_per_sample_lpips(x, x_adv, lpips_fn)

            all_clean_masks.append(c_mask)
            all_fooled_masks.append(fooled_mask)
            all_l0s.append(l0_per)
            all_l2s.append(l2_per)
            all_linfs.append(linf_per)
            all_psnrs.append(psnr_per)
            all_ssims.append(ssim_per)
            if lpips_per is not None:
                all_lpipss.append(lpips_per)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        c_mask_tensor = torch.cat(all_clean_masks)
        fooled_mask_tensor = torch.cat(all_fooled_masks)
        l0_tensor = torch.cat(all_l0s)
        l2_tensor = torch.cat(all_l2s)
        linf_tensor = torch.cat(all_linfs)
        psnr_tensor = torch.cat(all_psnrs)
        ssim_tensor = torch.cat(all_ssims)
        lpips_tensor = torch.cat(all_lpipss) if all_lpipss else None

        clean_correct = c_mask_tensor.sum().item()
        total_count = l0_tensor.numel()
        clean_acc = 100.0 * clean_correct / total_count

        for K in K_VALUES:
            # Correct Group B Mask: Attack MUST succeed AND modified pixels <= K
            succ_k_mask = fooled_mask_tensor & (l0_tensor <= K)
            success_count_k = int(succ_k_mask.sum().item())

            robust_acc_k = 100.0 * (clean_correct - success_count_k) / total_count
            cond_asr_k = 100.0 * success_count_k / max(1, clean_correct)

            dist_m_k = compute_distortion_metrics(
                l0_tensor, l2_tensor, linf_tensor, psnr_tensor, ssim_tensor, lpips_tensor, succ_k_mask
            )

            res = {
                "Group": "Group B", "Attack Method": name, "K": K,
                "Implementation Source": source_label,
                "Clean Acc (%)": round(clean_acc, 2), 
                "Robust Acc (%)": round(robust_acc_k, 2),
                "Conditional ASR (%)": round(cond_asr_k, 2), 
                "Accuracy Drop (%)": round(clean_acc - robust_acc_k, 2),
                "All Avg L0": round(dist_m_k["all_l0_mean"], 2),
                "Success Avg L0": round(dist_m_k["succ_l0_mean"], 2), 
                "Success Median L0": round(dist_m_k["succ_l0_median"], 2),
                "Success Avg L2": round(dist_m_k["succ_l2_mean"], 4), 
                "Success Avg L_inf": round(dist_m_k["succ_linf_mean"], 4),
                "Success PSNR (dB)": round(dist_m_k["succ_psnr_mean"], 2), 
                "Success SSIM": round(dist_m_k["succ_ssim_mean"], 4),
                "Success LPIPS": round(dist_m_k["succ_lpips_mean"], 4) if dist_m_k["succ_lpips_mean"] is not None else None,
                "Avg Iterations": round(total_steps / total_count, 2),
                "Time/Img (s)": round(dt / total_count, 4)
            }
            logger.info(f"[Group B] {name} (K={K}): Conditional ASR@K={res['Conditional ASR (%)']}%, Robust Acc={res['Robust Acc (%)']}%, Success Avg L0={res['Success Avg L0']}")
            results_list.append(res)
            pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    df_all = pd.DataFrame(results_list)
    df_all.to_csv(full_csv_path, index=False)
    
    json_records = df_all.where(pd.notnull(df_all), None).to_dict(orient="records")
    with open(os.path.join(METRICS_DIR, f"{output_prefix}_full_attack_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=4)

    return df_all

if __name__ == "__main__":
    logger.info("=== Running Group A, B, C Experimental Attack Benchmark ===")
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "local_test_10"
    use_official = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    
    logger.info(f"Benchmark configured with num_samples={num_samples}, output_prefix='{output_prefix}', use_official_adapters={use_official}")
    model = get_model("resnet18", pretrained=False)
    ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if ckpt:
        model = get_model(checkpoint_path=ckpt, device=DEVICE)
        
    _, _, test_loader = get_dataloaders(batch_size=EVAL_BATCH_SIZE)
    df_results = run_attack_benchmark_suite(
        model, 
        test_loader, 
        eval_batch_size=EVAL_BATCH_SIZE, 
        num_samples=num_samples, 
        device=DEVICE, 
        use_official_adapters=use_official,
        output_prefix=output_prefix
    )
    logger.info(f"\n{df_results.to_string(index=False)}")
