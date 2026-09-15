# Sparse Adversarial Attack Benchmark (`aa`)

This repository provides a minimal, reproducible research benchmark suite for pixel-sparse adversarial attacks on deep neural network classifiers under the spatial $L_0$ threat model.

## Architecture

```text
.
├── pyproject.toml                     # Editable package configuration (aa)
├── README.md                          # Quick start guide & system overview
├── THIRD_PARTY.md                     # Scientific provenance & upstream baselines
│
├── configs/
│   ├── paper_cifar10.yaml             # Official paper benchmark configuration (SOTA Iteration 5)
│   ├── paper.yaml                     # Canonical reproducible paper experiment configuration
│   ├── casa.yaml                      # Dedicated CASA SOTA evaluation configuration
│   └── smoke.yaml                     # Fast smoke test configuration for CI/CD
│
├── docs/
│   ├── proposed_method.md             # Formulation & algorithmic specification of CASA (SOTA Iter 5)
│   ├── benchmark_results_cifar10.md   # Official benchmark results & 11-attack comparison on CIFAR-10
│   ├── proposed_defense.md            # Defense strategies (Preprocessing filters, BPDA, & CASA-AT)
│   └── experiments_and_baselines.md   # Research scope, configs, & 10 competitor baseline guides
│
├── src/
│   └── aa/
│       ├── attacks/                   # Base class, dense baselines, CASA SOTA, registry, & external adapters
│       ├── benchmark.py               # Single generic evaluation loop
│       ├── data.py                    # Stratified CIFAR-10 data loaders & sample selection
│       ├── defenses.py                # Preprocessing defenses (Blur, Median, JPEG, TVM) & BPDA adapter
│       ├── metrics.py                 # Spatial L0, exact top-K, projections, and image quality metrics
│       ├── models.py                  # ResNet-18 & WRN-28-10 backbones with checkpoint integrity checks
│       ├── training/                  # Clean & Adversarial training engines
│       └── utils.py                   # Seed, device, and hash reproducibility utilities
│
├── scripts/
│   ├── attack_benchmark.py            # CLI runner for comprehensive multi-baseline benchmark
│   ├── defense_benchmark.py           # CLI runner for defense evaluation (oblivious & adaptive BPDA)
│   ├── run_casa_benchmark.py          # High-performance runner for CASA SOTA benchmark
│   ├── evaluate_checkpoint.py         # Checkpoint SHA256 integrity and accuracy verification
│   └── generate_markdown_report.py    # Report generator from JSON benchmark results
│
├── tests/                             # PyTest unit, contract, and benchmark verification tests
└── third_party/                       # Upstream official author implementations
```

## Documentation

- [Proposed Method (CASA)](docs/proposed_method.md) — Unified mathematical formulation of Coalition-Aware Sparse Adversarial Attack (Iteration 5).
- [Benchmark Results (CIFAR-10)](docs/benchmark_results_cifar10.md) — Official results on 1,000 CIFAR-10 samples comparing CASA against 10 baselines.
- [Proposed Defense Strategies](docs/proposed_defense.md) — Preprocessing filters (Median 3x3, TVM, JPEG), BPDA adaptive testing, and Sparse Adversarial Training (CASA-AT).
- [Experiments, Baselines & Protocol](docs/experiments_and_baselines.md) — Scope (ResNet-18, ResNet-50, CIFAR-10/100), configuration hierarchy, and execution instructions for all 10 competitor baselines.
- [Third-Party Provenance](THIRD_PARTY.md) — Upstream repository links, pinned commit SHAs, and adapter mapping.

## Quick Start

### 1. Installation

```bash
pip install -e .
```

### 2. Verify Model Checkpoint Integrity

```bash
python scripts/evaluate_checkpoint.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint result/saved_models/resnet18_cifar10_best.pth \
    --expected-sha256 378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172
```

### 3. Run Benchmark Suite

Run full multi-attack benchmark:
```bash
python scripts/attack_benchmark.py --config configs/paper_cifar10.yaml
```

Or run the proposed CASA SOTA attack:
```bash
python scripts/run_casa_benchmark.py --k 1 2 4 8 16 32 64 --batch-size 16 --samples 1000
```

### 4. Run Defense Benchmark

```bash
python scripts/defense_benchmark.py --config configs/paper_cifar10.yaml --defense median --mode adaptive
```

### 5. Run Test Suite

```bash
pytest tests/
```
