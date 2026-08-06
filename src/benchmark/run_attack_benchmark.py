# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import time
import math
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd

try:
    import lpips
except ImportError:
    lpips = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.datasets.dataset_loader import get_dataloaders
from src.models.model_factory import get_model, find_existing_checkpoint

# Import baseline & non-pixel-K attacks (Group C)
from src.attacks.baselines.fgsm import FGSMAttack
from src.attacks.baselines.bim import BIMAttack
from src.attacks.baselines.pgd import PGDAttack
from src.attacks.frequency.sfa import SpectralFrequencyAttack

# Import direct K-sweep attacks (Group A)
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
# IMAGE QUALITY METRICS ENGINE
# ==============================================================================
def compute_psnr(orig, adv):
    mse = torch.mean((orig - adv) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()

def compute_ssim(orig, adv):
    kernel = torch.ones((3, 1, 3, 3), device=orig.device) / 9.0
    mu_x = F.conv2d(orig, kernel, padding=1, groups=3)
    mu_y = F.conv2d(adv, kernel, padding=1, groups=3)

    sigma_x_sq = F.conv2d(orig * orig, kernel, padding=1, groups=3) - mu_x * mu_x
    sigma_y_sq = F.conv2d(adv * adv, kernel, padding=1, groups=3) - mu_y * mu_y
    sigma_xy = F.conv2d(orig * adv, kernel, padding=1, groups=3) - mu_x * mu_y

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean().item()

# ==============================================================================
# EXPERIMENTAL BENCHMARK ENGINE (GROUPS A, B, C)
# ==============================================================================
def run_attack_benchmark_suite(model, test_loader, eval_batch_size=EVAL_BATCH_SIZE, num_samples=NUM_BENCHMARK_TEST_SAMPLES, device=DEVICE):
    test_ds = test_loader.dataset
    if num_samples and num_samples < len(test_ds):
        logger.info(f"Subsetting test set to {num_samples} samples for benchmark evaluation...")
        test_ds = Subset(test_ds, range(num_samples))

    eval_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)
    logger.info(f"Evaluation DataLoader created with batch_size={eval_batch_size}, total_samples={len(test_ds)}")

    lpips_fn = None
    if lpips is not None:
        try:
            lpips_fn = lpips.LPIPS(net='alex').to(device)
        except Exception:
            lpips_fn = None

    K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
    results_list = []
    full_csv_path = os.path.join(METRICS_DIR, "full_attack_metrics.csv")

    # ==========================================================================
    # GROUP C: Non-pixel-K Attacks (Dense + Spectral Frequency Domain)
    # ==========================================================================
    logger.info("=== GROUP C: Non-pixel-K Attacks (FGSM, BIM, PGD, SFA) ===")
    group_c_attacks = {
        "FGSM": FGSMAttack(model, device=device),
        "BIM": BIMAttack(model, device=device),
        "PGD": PGDAttack(model, device=device),
        "SFA": SpectralFrequencyAttack(model, freq_k=15, device=device)
    }

    for name, attacker in group_c_attacks.items():
        t0 = time.time()
        clean_correct, total_count, robust_correct, adv_succ = 0, 0, 0, 0
        total_l0, total_l2, total_linf = 0.0, 0.0, 0.0
        total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
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
            diff = (x_adv - x).abs()
            l0_per = torch.sum(diff.max(dim=1)[0] > 1e-4, dim=(1, 2)).float()
            l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
            linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

            clean_correct += c_mask.sum().item()
            robust_correct += r_mask.sum().item()
            total_count += B
            adv_succ += (c_mask & (~r_mask)).sum().item()

            total_l0 += l0_per.sum().item()
            total_l2 += l2_per.sum().item()
            total_linf += linf_per.sum().item()

            total_psnr += compute_psnr(x, x_adv) * B
            total_ssim += compute_ssim(x, x_adv) * B
            if lpips_fn:
                with torch.no_grad():
                    total_lpips += lpips_fn(x * 2 - 1, x_adv * 2 - 1).mean().item() * B

        dt = time.time() - t0
        clean_acc = 100.0 * clean_correct / total_count
        rob_acc = 100.0 * robust_correct / total_count
        asr = 100.0 * adv_succ / max(1, clean_correct)

        res = {
            "Group": "Group C", "Attack Method": name, "K": "N/A",
            "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc, 2),
            "ASR (%)": round(asr, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc, 2),
            "Avg L0": round(total_l0 / total_count, 2), "Avg L0 Ratio": round((total_l0 / total_count) / 1024.0, 4),
            "Avg L2": round(total_l2 / total_count, 4), "Avg L_inf": round(total_linf / total_count, 4),
            "PSNR (dB)": round(total_psnr / total_count, 2), "SSIM": round(total_ssim / total_count, 4),
            "LPIPS": round(total_lpips / total_count, 4) if lpips_fn else float('nan'),
            "Avg Iterations": round(total_steps / total_count, 2),
            "Time/Img (s)": round(dt / total_count, 4)
        }
        logger.info(f"[Group C] {name}: ASR={res['ASR (%)']}%, Robust Acc={res['Robust Acc (%)']}%, Avg L0={res['Avg L0']}")
        results_list.append(res)
        pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    # ==========================================================================
    # GROUP A: Direct K-Sweep Attacks (Budget Constrained)
    # ==========================================================================
    logger.info("=== GROUP A: Direct K-Sweep Attacks (Explicit K-budget) ===")
    group_a_factories = {
        "JSMA": lambda m, k, d: JSMAAttack(m, k=k, device=d),
        "OnePixel": lambda m, k, d: OnePixelAttack(m, k=k, device=d),
        "CornerSearch": lambda m, k, d: CornerSearchAttack(m, k=k, device=d),
        "SAIF": lambda m, k, d: SAIFAttack(m, k=k, device=d),
        "PGD0": lambda m, k, d: PGD0Attack(m, k=k, device=d),
        "Sparse-PGD": lambda m, k, d: SparsePGDAttack(m, sparsity_budget=k, device=d),
        "Sparse-RS": lambda m, k, d: SparseRSAttack(m, n_pixels=k, device=d),
        "BruSLe": lambda m, k, d: BruSLeAttack(m, block_size=max(1, int(k**0.5)), device=d),
        "IPFSA": lambda m, k, d: IPFSAttack(m, k_pixels=k, device=d),
        "GradientGuidance": lambda m, k, d: GradientGuidanceAttack(m, sparsity_budget=k, device=d),
        "CPA": lambda m, k, d: CooperativePixelsAttack(m, coalition_size=k, device=d),
        "FCSA": lambda m, k, d: FunctionalCoalitionSparseAttack(m, max_coalition_size=k, device=d),
        "FMSA-budgeted": lambda m, k, d: FeatureToMinimalSupportAttack(m, support_budget=k, device=d),
        "HSA-budgeted": lambda m, k, d: HypergraphSparseAttack(m, budget=k, device=d)
    }

    for name, factory in group_a_factories.items():
        for K in K_VALUES:
            attacker = factory(model, K, device)
            t0 = time.time()
            clean_correct, total_count, robust_correct, adv_succ = 0, 0, 0, 0
            total_l0, total_l2, total_linf = 0.0, 0.0, 0.0
            total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
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
                diff = (x_adv - x).abs()
                l0_per = torch.sum(diff.max(dim=1)[0] > 1e-4, dim=(1, 2)).float()
                l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
                linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

                clean_correct += c_mask.sum().item()
                robust_correct += r_mask.sum().item()
                total_count += B
                adv_succ += (c_mask & (~r_mask)).sum().item()

                total_l0 += l0_per.sum().item()
                total_l2 += l2_per.sum().item()
                total_linf += linf_per.sum().item()

                total_psnr += compute_psnr(x, x_adv) * B
                total_ssim += compute_ssim(x, x_adv) * B
                if lpips_fn:
                    with torch.no_grad():
                        total_lpips += lpips_fn(x * 2 - 1, x_adv * 2 - 1).mean().item() * B

            dt = time.time() - t0
            clean_acc = 100.0 * clean_correct / total_count
            rob_acc = 100.0 * robust_correct / total_count
            asr = 100.0 * adv_succ / max(1, clean_correct)
            avg_l0_val = round(total_l0 / total_count, 2)

            res = {
                "Group": "Group A", "Attack Method": name, "K": K,
                "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc, 2),
                "ASR (%)": round(asr, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc, 2),
                "Avg L0": avg_l0_val, "Avg L0 Ratio": round(avg_l0_val / 1024.0, 4),
                "Avg L2": round(total_l2 / total_count, 4), "Avg L_inf": round(total_linf / total_count, 4),
                "PSNR (dB)": round(total_psnr / total_count, 2), "SSIM": round(total_ssim / total_count, 4),
                "LPIPS": round(total_lpips / total_count, 4) if lpips_fn else float('nan'),
                "Avg Iterations": round(total_steps / total_count, 2),
                "Time/Img (s)": round(dt / total_count, 4)
            }
            logger.info(f"[Group A] {name} (K={K}): ASR={res['ASR (%)']}%, Robust Acc={res['Robust Acc (%)']}%, Avg L0={res['Avg L0']}")
            results_list.append(res)
            pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    # ==========================================================================
    # GROUP B: Unconstrained Minimum Support Optimization -> Calculate ASR@K
    # ==========================================================================
    logger.info("=== GROUP B: Minimal Support Optimization (Post-hoc ASR@K Evaluation) ===")
    group_b_attacks = {
        "SparseFool": SparseFoolAttack(model, k=250, steps=50, device=device),
        "SigmaZero": SigmaZeroAttack(model, steps=50, device=device),
        "Homotopy": HomotopyAttack(model, target_sparsity=250, steps=50, device=device),
        "GSE": GroupSparseAttack(model, group_size=4, max_groups=64, steps=50, device=device),
        "Pixle": PixleAttack(model, n_swaps=20, max_trials=50, device=device),
        "FMSA-minimal-support": FeatureToMinimalSupportAttack(model, support_budget=250, device=device)
    }

    for name, attacker in group_b_attacks.items():
        t0 = time.time()
        clean_correct, total_count = 0, 0
        sample_l0s, sample_l2s, sample_linfs = [], [], []
        sample_psnrs, sample_ssims, sample_lpipss = [], [], []
        sample_fooled = []
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
            diff = (x_adv - x).abs()
            l0_per = torch.sum(diff.max(dim=1)[0] > 1e-4, dim=(1, 2)).float()
            l2_per = torch.norm(diff.view(B, -1), p=2, dim=1)
            linf_per = torch.norm(diff.view(B, -1), p=float('inf'), dim=1)

            clean_correct += c_mask.sum().item()
            total_count += B

            sample_fooled.extend(fooled_mask.cpu().numpy().tolist())
            sample_l0s.extend(l0_per.cpu().numpy().tolist())
            sample_l2s.extend(l2_per.cpu().numpy().tolist())
            sample_linfs.extend(linf_per.cpu().numpy().tolist())

        dt = time.time() - t0
        clean_acc = 100.0 * clean_correct / total_count

        fooled_arr = pd.Series(sample_fooled)
        l0_arr = pd.Series(sample_l0s)
        l2_arr = pd.Series(sample_l2s)
        linf_arr = pd.Series(sample_linfs)

        for K in K_VALUES:
            succ_k = fooled_arr & (l0_arr <= K)
            asr_k = 100.0 * succ_k.sum() / max(1, clean_correct)
            rob_acc_k = clean_acc - asr_k

            res = {
                "Group": "Group B", "Attack Method": name, "K": K,
                "Clean Acc (%)": round(clean_acc, 2), "Robust Acc (%)": round(rob_acc_k, 2),
                "ASR (%)": round(asr_k, 2), "Accuracy Drop (%)": round(clean_acc - rob_acc_k, 2),
                "Avg L0": round(l0_arr.mean(), 2), "Avg L0 Ratio": round(l0_arr.mean() / 1024.0, 4),
                "Avg L2": round(l2_arr.mean(), 4), "Avg L_inf": round(linf_arr.mean(), 4),
                "PSNR (dB)": float('nan'), "SSIM": float('nan'), "LPIPS": float('nan'),
                "Avg Iterations": round(total_steps / total_count, 2),
                "Time/Img (s)": round(dt / total_count, 4)
            }
            logger.info(f"[Group B] {name} (K={K}): ASR@K={res['ASR (%)']}%, Mean L0={res['Avg L0']}")
            results_list.append(res)
            pd.DataFrame(results_list).to_csv(full_csv_path, index=False)

    df_all = pd.DataFrame(results_list)
    df_all.to_csv(full_csv_path, index=False)
    with open(os.path.join(METRICS_DIR, "full_attack_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(df_all.to_dict(orient="records"), f, indent=4)

    return df_all

if __name__ == "__main__":
    logger.info("=== Running Group A, B, C Experimental Attack Benchmark ===")
    model = get_model("resnet18", pretrained=False)
    ckpt = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if ckpt:
        model = get_model(checkpoint_path=ckpt, device=DEVICE)
    _, _, test_loader = get_dataloaders(batch_size=EVAL_BATCH_SIZE)
    df_results = run_attack_benchmark_suite(model, test_loader, eval_batch_size=EVAL_BATCH_SIZE, num_samples=NUM_BENCHMARK_TEST_SAMPLES, device=DEVICE)
    logger.info(f"\n{df_results.to_string(index=False)}")
