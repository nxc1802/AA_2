#!/usr/bin/env python3
"""
run_on_kaggle.py — Self-Contained Kaggle Benchmark Runner & Bootstrap Script
=============================================================================
This script can be executed in terminal OR copied & pasted directly into a single
Kaggle Notebook cell to run the entire CASA paper-ready benchmark suite.

Features:
1. Auto-clones https://github.com/nxc1802/AA_2.git if not already present.
2. Auto-installs dependencies and package in editable mode.
3. Verifies ResNet-18 clean checkpoint via SHA256 checksum (auto-downloads if missing).
4. Executes desired stages:
   - smoke:        Sanity test on 20 samples (< 1 min)
   - casa_10k:     Official 10,000-sample CIFAR-10 test set benchmark for CASA
   - casa_1k:      Fast 1,000-sample validation for CASA
   - baselines_10k: SPGD, Sigma-Zero, Sparse-RS on 10,000 test set
   - ablation:     Component ablation study on 1,000 samples (K=1, 4, 16)
   - failures:     Diagnostic failure case analysis on K=1, 2, 4
   - plot:         Render all 4 publication-quality figures (300 DPI)
   - all:          Execute all stages sequentially
5. Auto-packages all artifacts into paper_artifacts.tar.gz in /kaggle/working/ for 1-click download.
"""

import os
import sys
import shutil
import hashlib
import argparse
import subprocess


REPO_URL = "https://github.com/nxc1802/AA_2.git"
REPO_DIR = "AA_2"
EXPECTED_SHA256 = "378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172"
CHECKPOINT_PATH = "result/saved_models/resnet18_cifar10_best.pth"


def run_command(cmd, desc=None, check=True):
    if desc:
        print(f"\n{'='*70}\n--> {desc}\n{'='*70}", flush=True)
    print(f"[EXEC]: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if check and res.returncode != 0:
        print(f"❌ Error: Command failed with exit code {res.returncode}: {cmd}", file=sys.stderr)
        sys.exit(res.returncode)
    return res.returncode


def setup_environment():
    """Clones repo if needed, enters repo directory, installs dependencies."""
    print("=" * 70)
    print(" 🚀 CASA Kaggle Benchmark Bootstrap")
    print("=" * 70, flush=True)

    # 1. Detect if inside repo or in Kaggle working dir
    curr_dir = os.getcwd()
    if not os.path.exists("pyproject.toml"):
        if not os.path.exists(REPO_DIR):
            run_command(f"git clone {REPO_URL}", desc="Cloning Repository from GitHub")
        else:
            print(f"Repository {REPO_DIR} already exists.")
        
        os.chdir(REPO_DIR)
        print(f"Changed working directory to: {os.getcwd()}", flush=True)

    # Ensure src is in sys.path
    src_abs = os.path.abspath("src")
    if src_abs not in sys.path:
        sys.path.insert(0, src_abs)

    # 2. Hardware diagnostic
    print("\n--- Hardware & PyTorch Diagnostics ---", flush=True)
    run_command("nvidia-smi || true", check=False)
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"CUDA Devices:    {count} device(s)")
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  [GPU {i}]: {name} ({mem:.2f} GB VRAM)")

    # 3. Dependencies installation
    run_command("pip install -q datasets huggingface_hub lpips pyyaml", desc="Installing Core Dependencies")
    run_command("pip install -q -e .", desc="Installing AA Package (Editable Mode)")

    # 4. Checkpoint Verification
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nCheckpoint not found locally. Downloading from Hugging Face Hub (Cuong2004/AA)...", flush=True)
        from aa.models import find_existing_checkpoint
        find_existing_checkpoint(CHECKPOINT_PATH)

    with open(CHECKPOINT_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()

    assert actual_sha == EXPECTED_SHA256, (
        f"Checksum error! Expected {EXPECTED_SHA256}, got {actual_sha}"
    )
    print(f"✅ Checkpoint verified successfully: {CHECKPOINT_PATH}")
    print(f"   SHA256: {actual_sha}", flush=True)


def execute_pipeline(stage: str = "all", batch_size: int = 16):
    """Executes chosen benchmark stages."""
    os.makedirs("result", exist_ok=True)

    # Stage 0: Smoke Test
    if stage in ("smoke", "all"):
        run_command(
            "python scripts/attack_benchmark.py --config configs/smoke.yaml --strict --output result/smoke_results.json",
            desc="[STAGE 0] Running Sanity Smoke Test (20 samples)"
        )

    # Stage 1: CASA 10k Official Test Set
    if stage in ("casa_10k", "all"):
        run_command(
            f"python scripts/run_casa_benchmark.py --samples 10000 --batch-size {batch_size} --k-values 1 2 4 8 16 32 64 --output result/casa_10000_results.json",
            desc="[STAGE 1] Running CASA Official 10,000-Sample CIFAR-10 Benchmark"
        )
    elif stage == "casa_1k":
        run_command(
            f"python scripts/run_casa_benchmark.py --samples 1000 --batch-size {batch_size} --k-values 1 2 4 8 16 32 64 --output result/casa_1000_results.json",
            desc="[STAGE 1] Running CASA Fast 1,000-Sample Validation Benchmark"
        )

    # Stage 2: Competitor Baselines (SPGD, Sigma-Zero, Sparse-RS)
    if stage in ("baselines_10k", "all"):
        run_command(
            "python scripts/attack_benchmark.py --config configs/paper_cifar10.yaml --attacks spgd,sigma_zero --output result/baselines_spgd_sigmazero_10k.json",
            desc="[STAGE 2] Running Whitebox SOTA Baselines (SPGD & Sigma-Zero) on 10,000 Samples"
        )
        run_command(
            "python scripts/attack_benchmark.py --config configs/paper_cifar10.yaml --attacks sparse_rs --output result/sparse_rs_10k.json",
            desc="[STAGE 2] Running Blackbox Baseline (Sparse-RS 10k queries) on 10,000 Samples"
        )

    # Stage 3: Component Ablation Study
    if stage in ("ablation", "all"):
        run_command(
            "python scripts/run_ablation.py --samples 1000 --k-values 1 4 16 --output result/ablation_results.json --report-md docs/ablation_study.md",
            desc="[STAGE 3] Running CASA Component Ablation Study (1,000 samples, K=1, 4, 16)"
        )

    # Stage 4: Diagnostic Failure Analysis
    if stage in ("failures", "all"):
        run_command(
            "python scripts/analyze_failures.py --samples 1000 --k-values 1 2 4 --max-failures 30 --output-json result/failure_analysis.json --output-md docs/failure_analysis.md",
            desc="[STAGE 4] Running Diagnostic Failure Case Analysis"
        )

    # Stage 5: Plot Paper Figures
    if stage in ("plot", "all"):
        run_command(
            "python scripts/plot_paper_figures.py --output-dir result/figures",
            desc="[STAGE 5] Rendering High-Resolution Paper Figures (300 DPI)"
        )

    # Package Artifacts for 1-click Download
    package_artifacts()


def package_artifacts():
    print(f"\n{'='*70}\n--> Packaging Artifacts for Download\n{'='*70}", flush=True)
    archive_name = "paper_artifacts_cifar10.tar.gz"
    
    # Create archive
    run_command(f"tar -czf {archive_name} result/ docs/ configs/", check=False)

    # If running on Kaggle, copy to /kaggle/working/ so it appears in the right pane
    kaggle_out = "/kaggle/working"
    if os.path.exists(kaggle_out) and os.path.abspath(kaggle_out) != os.path.abspath("."):
        dest_archive = os.path.join(kaggle_out, archive_name)
        shutil.copy2(archive_name, dest_archive)
        print(f"📦 Successfully copied artifact archive to Kaggle Output: {dest_archive}", flush=True)

    print(f"\n🎉 ALL TASKS COMPLETE! Artifacts packaged into: {archive_name}")
    print(f"   You can download this file directly from the Kaggle Output panel.\n")


def main():
    parser = argparse.ArgumentParser(description="Automated Kaggle Benchmark Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["smoke", "casa_10k", "casa_1k", "baselines_10k", "ablation", "failures", "plot", "all"],
        help="Benchmark stage to execute (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for CASA attack (default: 16, or 32 for Tesla T4)"
    )
    args = parser.parse_args()

    setup_environment()
    execute_pipeline(stage=args.stage, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
