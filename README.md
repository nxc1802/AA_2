# Sparse Adversarial Attack Benchmark (AA_2)

This repository provides a standardized research benchmark suite for pixel-sparse adversarial attacks on deep neural network classifiers.

## Architecture

```
.
├── src/
│   ├── attacks/            # Attack implementations (baselines, classical, blackbox, proposed, adapters)
│   ├── benchmark/          # Benchmark suite runner scripts (attack & defense)
│   ├── core/               # Spatial L0 projections, distortion metrics, and utils
│   ├── datasets/           # PyTorch & Hugging Face dataset loaders
│   ├── defenses/           # Pre-processing defense modules (Blur, Median, JPEG, TVM)
│   ├── models/             # ResNet-18/50 backbones and HF model checkpoint manager
│   ├── reports/            # Automated metric formatting
│   └── visualization/      # Plotting and curve visualization scripts
├── tests/                  # PyTest contract & scientific correctness tests
├── third_party/            # Official author baseline implementations
└── result/                 # Execution metrics, outputs, and log files
```

## Key Features

1. **Exact Spatial $L_0$ Definition**:
   Calculates spatial pixel modification $L_0 \in [0, H \times W]$ where a pixel is considered modified if any color channel changes beyond $\epsilon$:
   $$\text{modified\_pixels} = \sum_{h,w} \mathbb{I}\left(\max_{c} |\Delta_{c,h,w}| > \epsilon\right)$$

2. **Calibrated Metrics**:
   Outputs both **All-sample metrics** and **Success-conditioned metrics** (Avg $L_0$, Median $L_0$, $L_2$, $L_\infty$, PSNR, SSIM, LPIPS) computed strictly over successful adversarial perturbations to prevent failed attacks ($L_0=0$) from distorting quality evaluations.

3. **Multi-Group Evaluation**:
   - **Group A**: Budget-constrained $K$-sweep ($K \in \{1, 2, 4, 8, 16, 32, 64, 128\}$)
   - **Group B**: Minimal-support unconstrained attacks evaluated via cumulative ASR@K
   - **Group C**: Dense $L_\infty$ / Spectral frequency domain baselines (FGSM, BIM, PGD, SFA)

4. **Four Proposed Sparse Attack Methods**:
   - **CPA**: Cooperative Pixels Attack (local spatial neighborhood saliency cooperation)
   - **FCSA**: Functional Coalition Sparse Attack (joint RGB channel & spatial coalition scoring)
   - **FMSA**: Feature-to-Minimal Support Attack (joint CE + feature representation disruption loss)
   - **HSA**: Hypergraph Sparse Attack (multi-scale spatial receptive field hyperedge pooling)

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Local 10-Sample Benchmark Test

```bash
python src/benchmark/run_attack_benchmark.py 10 local_test_10
```

### 3. Run Defense Benchmark

```bash
python src/benchmark/run_defense_benchmark.py 50
```

### 4. Run Unit Tests

```bash
pytest tests/
```
