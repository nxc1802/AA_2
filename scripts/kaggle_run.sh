#!/usr/bin/env bash
# ==============================================================================
# scripts/kaggle_run.sh — Automated Master Execution Script for Kaggle GPU
# ==============================================================================
set -e

STAGE=${1:-"all"}
DEVICE=${2:-"cuda"}

echo "=============================================================================="
echo " CASA Paper Benchmark Runner (Kaggle Environment)"
echo " Stage: $STAGE | Device: $DEVICE"
echo "=============================================================================="

# 1. Environment & GPU Diagnostic
nvidia-smi || echo "Warning: nvidia-smi failed, running on CPU or unsupported driver."
python3 -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device Count:', torch.cuda.device_count())"

# 2. Install dependencies & package in editable mode
pip install -r requirements.txt --quiet
pip install -e . --quiet

# 3. Checkpoint Verification
CHECKPOINT_PATH="result/saved_models/resnet18_cifar10_best.pth"
EXPECTED_SHA256="378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found locally at $CHECKPOINT_PATH. Downloading from Hugging Face Hub (Cuong2004/AA)..."
    python3 -c "from aa.models import find_existing_checkpoint; find_existing_checkpoint('$CHECKPOINT_PATH')"
fi

echo "Verifying Checkpoint Checksum..."
ACTUAL_SHA256=$(sha256sum "$CHECKPOINT_PATH" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: SHA256 checksum mismatch!"
    echo "Expected: $EXPECTED_SHA256"
    echo "Actual:   $ACTUAL_SHA256"
    exit 1
fi
echo "Checkpoint SHA256 matches verified paper standard."

# ------------------------------------------------------------------------------
# STAGE 0: Quick Smoke Verification (20 samples)
# ------------------------------------------------------------------------------
if [ "$STAGE" == "smoke" ] || [ "$STAGE" == "all" ]; then
    echo ""
    echo "--> [STAGE 0] Running Smoke Sanity Check (20 samples) ..."
    python3 scripts/attack_benchmark.py --config configs/smoke.yaml --strict --output result/smoke_test_results.json
fi

# ------------------------------------------------------------------------------
# STAGE 1: CASA Official SOTA 10k Benchmark
# ------------------------------------------------------------------------------
if [ "$STAGE" == "casa_10k" ] || [ "$STAGE" == "all" ]; then
    echo ""
    echo "--> [STAGE 1] Running CASA 10,000 Samples Official Benchmark ..."
    python3 scripts/run_casa_benchmark.py \
        --samples 10000 \
        --batch-size 16 \
        --k-values 1 2 4 8 16 32 64 \
        --output result/casa_10000_results.json
fi

# ------------------------------------------------------------------------------
# STAGE 2: Strong Competitor Baselines (SPGD, Sigma-Zero, Sparse-RS)
# ------------------------------------------------------------------------------
if [ "$STAGE" == "baselines_10k" ] || [ "$STAGE" == "all" ]; then
    echo ""
    echo "--> [STAGE 2] Running Strong Whitebox Baselines (SPGD & Sigma-Zero) on 10,000 Samples ..."
    python3 scripts/attack_benchmark.py \
        --config configs/paper_cifar10.yaml \
        --attacks spgd,sigma_zero \
        --output result/baselines_spgd_sigmazero_10k.json

    echo ""
    echo "--> Running Sparse-RS (10,000 queries) on 10,000 Samples ..."
    python3 scripts/attack_benchmark.py \
        --config configs/paper_cifar10.yaml \
        --attacks sparse_rs \
        --output result/sparse_rs_10k.json
fi

# ------------------------------------------------------------------------------
# STAGE 3: Component Ablation Study (1,000 Samples, K=1, 4, 16)
# ------------------------------------------------------------------------------
if [ "$STAGE" == "ablation" ] || [ "$STAGE" == "all" ]; then
    echo ""
    echo "--> [STAGE 3] Running CASA Component Ablation (1,000 samples) ..."
    python3 scripts/run_ablation.py \
        --samples 1000 \
        --k-values 1 4 16 \
        --output result/ablation_results.json \
        --report-md docs/ablation_study.md
fi

# ------------------------------------------------------------------------------
# STAGE 4: Diagnostic Failure Analysis (K=1, 2, 4)
# ------------------------------------------------------------------------------
if [ "$STAGE" == "failures" ] || [ "$STAGE" == "all" ]; then
    echo ""
    echo "--> [STAGE 4] Running Failure Case Diagnostics ..."
    python3 scripts/analyze_failures.py \
        --samples 1000 \
        --k-values 1 2 4 \
        --max-failures 30 \
        --output-json result/failure_analysis.json \
        --output-md docs/failure_analysis.md
fi

# ------------------------------------------------------------------------------
# STAGE 5: Generate Paper Figures & Final Markdown Report
# ------------------------------------------------------------------------------
if [ "$STAGE" == "report" ] || [ "$STAGE" == "all" ]; then
    echo ""
    echo "--> [STAGE 5] Generating Publication-Ready Figures & Reports ..."
    python3 scripts/plot_paper_figures.py --output-dir result/figures
    python3 scripts/generate_markdown_report.py \
        --results result/casa_10000_results.json result/baselines_spgd_sigmazero_10k.json result/sparse_rs_10k.json \
        --output docs/official_paper_10k_results.md || true
fi

echo ""
echo "=============================================================================="
echo " All requested stages completed successfully!"
echo " Artifacts saved under result/ and docs/"
echo "=============================================================================="
