import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 🛡️ CIFAR-10 Sparse Adversarial Attack Benchmark
        **High-Performance Vectorized PyTorch Evaluation Engine running on Marimo**

        - **Backbone:** ResNet-18 (adapted for CIFAR-10 32x32 spatial inputs)
        - **Dataset:** CIFAR-10 (HF `Cuong2004/AA`) with stratified 40k train / 10k val / 10k test split
        - **Model Checkpoint:** Auto-downloaded from Hugging Face (`Cuong2004/AA/models/resnet18_cifar10_best.pth`)
        - **Batch Size:** `1024` (Maximized for Ultra-Fast Parallel Execution)
        - **Experimental Groups:**
          - **Group A (Direct K-Sweep):** JSMA, OnePixel, CornerSearch, SAIF, PGD0, Sparse-PGD, Sparse-RS, BruSLe, IPFSA, GradientGuidance, CPA, FCSA, FMSA-budgeted, HSA-budgeted
          - **Group B (Minimal Support Optimization & ASR@K):** SparseFool, SigmaZero, Homotopy, GSE, Pixle, FMSA-minimal-support
          - **Group C (Non-pixel-K Attacks):** FGSM, BIM, PGD, SFA (Spectral Frequency Attack)
        """
    )
    return (mo,)


@app.cell
def _():
    import os
    import sys
    import time
    import torch
    import pandas as pd

    sys.path.append(os.path.abspath("."))

    from src.datasets.dataset_loader import get_dataloaders
    from src.models.model_factory import (
        get_model,
        evaluate_accuracy,
        train_clean_resnet18,
        find_existing_checkpoint,
    )
    from src.benchmark.run_attack_benchmark import (
        run_attack_benchmark_suite,
    )
    from src.reports.generate_report import generate_reports

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_BATCH_SIZE = 1024
    EVAL_BATCH_SIZE = 1024
    NUM_BENCHMARK_TEST_SAMPLES = 1000

    print(f"=== MARIMO BENCHMARK PIPELINE (Device: {DEVICE}, Batch Size: {EVAL_BATCH_SIZE}) ===")
    return (
        DEVICE,
        EVAL_BATCH_SIZE,
        NUM_BENCHMARK_TEST_SAMPLES,
        TRAIN_BATCH_SIZE,
        evaluate_accuracy,
        find_existing_checkpoint,
        generate_reports,
        get_dataloaders,
        get_model,
        os,
        pd,
        run_attack_benchmark_suite,
        sys,
        time,
        torch,
        train_clean_resnet18,
    )


@app.cell
def _(find_existing_checkpoint, get_dataloaders, get_model, torch, DEVICE):
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=1024)
    ckpt_path = find_existing_checkpoint("resnet18_cifar10_best.pth")
    if ckpt_path:
        model = get_model(checkpoint_path=ckpt_path, device=DEVICE)
        print(f"Loaded ResNet-18 model from: {ckpt_path}")
    else:
        print("No checkpoint found on HF/locally. Use benchmark button to train and upload.")
        model = None
    return ckpt_path, model, test_loader, train_loader, val_loader


@app.cell
def _(mo):
    run_button = mo.ui.button(label="🚀 Run Benchmark Suite (Batch Size 1024)", value=False)
    run_button
    return (run_button,)


@app.cell
def _(
    DEVICE,
    NUM_BENCHMARK_TEST_SAMPLES,
    TRAIN_BATCH_SIZE,
    generate_reports,
    get_dataloaders,
    model,
    mo,
    run_attack_benchmark_suite,
    run_button,
    test_loader,
    train_clean_resnet18,
):
    mo.stop(not run_button.value, mo.md("Click **Run Benchmark Suite** above to start evaluation."))
    mo.status.toast(title="Running Benchmark", description="⏳ Starting CIFAR-10 Sparse Attack Benchmark...", kind="info")
    if model is not None:
        eval_model = model
        te_loader = test_loader
    else:
        tr_loader, v_loader, te_loader = get_dataloaders(batch_size=TRAIN_BATCH_SIZE)
        eval_model = train_clean_resnet18(tr_loader, v_loader, te_loader, epochs=200, device=DEVICE)
    df_results = run_attack_benchmark_suite(eval_model, te_loader, eval_batch_size=1024, num_samples=NUM_BENCHMARK_TEST_SAMPLES, device=DEVICE)
    generate_reports(df_results)
    mo.status.toast(title="Finished", description="🎉 Benchmark Completed Successfully!", kind="success")
    benchmark_display = mo.vstack([
        mo.md("### 📊 CIFAR-10 Sparse Attack Benchmark Results"),
        mo.ui.table(df_results)
    ])

    benchmark_display
    return benchmark_display, eval_model, te_loader


if __name__ == "__main__":
    app.run()
