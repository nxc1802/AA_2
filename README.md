# Sparse Adversarial Attack Benchmark (`aa`)

This repository provides a minimal, reproducible research benchmark suite for pixel-sparse adversarial attacks on deep neural network classifiers.

## Architecture

```text
.
├── pyproject.toml          # Editable package configuration (aa)
├── README.md               # Quick start guide
├── THIRD_PARTY.md          # Scientific provenance & upstream baselines
│
├── configs/
│   └── paper.yaml          # Reproducible paper experiment configuration
│
├── docs/
│   ├── protocol.md         # Single source of truth for experimental protocol
│   ├── proposed_method.md  # Formulation & design of SparseFeatureAttack
│   └── roadmap.md          # Project development roadmap & checklist
│
├── src/
│   └── aa/
│       ├── attacks/        # Base class, dense baselines, proposed method, registry, & external adapters
│       ├── benchmark.py    # Single generic evaluation loop
│       ├── data.py         # Stratified CIFAR-10 data loaders & sample selection
│       ├── defenses.py     # Preprocessing defenses (Blur, Median, JPEG, TVM) & BPDA adapter
│       ├── metrics.py      # Spatial L0, exact top-K, projections, and image quality metrics
│       ├── models.py       # ResNet-18 & WRN-28-10 backbones with HF checkpoint loading
│       └── utils.py        # Seed, device, and hash reproducibility utilities
│
├── scripts/
│   ├── attack_benchmark.py # CLI runner for attack benchmark
│   └── defense_benchmark.py# CLI runner for defense benchmark
│
├── tests/                  # PyTest contract, unit, and benchmark tests
└── third_party/            # Upstream official author implementations
```

## Documentation

- [Experimental Protocol](docs/protocol.md)
- [Proposed Method](docs/proposed_method.md)
- [Development Roadmap](docs/roadmap.md)
- [Third-Party Sources](THIRD_PARTY.md)

## Quick Start

### 1. Installation

```bash
pip install -e .
```

### 2. Run Attack Benchmark

```bash
python scripts/attack_benchmark.py --config configs/paper.yaml
```

### 3. Run Defense Benchmark

```bash
python scripts/defense_benchmark.py --config configs/paper.yaml
```

### 4. Run Unit Tests

```bash
pytest tests/
```
