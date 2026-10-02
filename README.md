# Sparse Adversarial Attack Benchmark (`aa`)

This repository provides a minimal, reproducible research benchmark suite for pixel-sparse adversarial attacks on deep neural network classifiers under the spatial $L_0$ threat model, featuring **CASA** (*Coalition-Aware Sparse Adversarial Attack*).

---

## 1. System Architecture

```text
.
├── pyproject.toml                     # Editable package configuration (aa)
├── README.md                          # Repository quickstart & entry point
├── LICENSE                            # MIT License
├── THIRD_PARTY.md                     # Scientific provenance & upstream baseline commits
│
├── configs/
│   ├── paper_cifar10.yaml             # Official paper benchmark configuration (10,000 samples)
│   ├── development.yaml               # Intermediate validation configuration (1,000 samples)
│   ├── ablation.yaml                  # Dedicated component ablation configuration
│   └── smoke.yaml                     # Fast smoke test configuration for CI/CD (20 samples)
│
├── docs/                              # Canonical 14-document research system
│   ├── 01_research_overview.md        # Research map, problem formulation, RQs & scope
│   ├── 02_threat_model_and_protocol.md# SSOT: Spatial L0, metrics (ASR, CRA) & budget protocol
│   ├── 03_casa_method.md              # Mathematical specification of CASA & code mapping
│   ├── 04_datasets.md                 # Dataset specifications, preprocessing & fingerprints
│   ├── 05_models_and_training.md      # Target model matrix, acceptance gate & checkpoint SHA256
│   ├── 06_baselines.md                # 8 competitor baselines, adapter methodology & rationale
│   ├── 07_experiment_plan.md          # Operational plan, P0–P5 roadmap & Go/No-Go gates
│   ├── 08_attack_benchmark.md         # Empirical benchmark results, tables & Pareto frontier
│   ├── 09_ablation_and_analysis.md    # 10-variant ablation, synergy study & failure analysis
│   ├── 10_defense_research.md         # Preprocessing defenses, adaptive BPDA & masking diagnostics
│   ├── 11_reproducibility.md          # 10-step reproduction cookbook & Kaggle GPU execution
│   ├── 12_artifacts_and_results.md    # Artifact index, raw JSON schemas & SHA256 sums
│   ├── 13_limitations.md              # Boundary conditions, method limits & negative scope
│   ├── 14_publication_checklist.md    # Definition of Done (DoD) & publication release gates
│   └── assets/                        # Publication figures (PNG, 300 DPI)
│
├── src/aa/
│   ├── attacks/                       # Modular CASA, dense baselines, registry & external adapters
│   │   └── casa/                      # First-class Support, Objective, Candidate, NMS, Compression
│   ├── defenses/                      # Preprocessing filters (Blur, Median, JPEG, TVM) & BPDA adapter
│   ├── benchmark.py                   # Single generic evaluation loop
│   ├── data.py                        # Stratified CIFAR data loaders & sample selection
│   ├── metrics.py                     # Spatial L0, exact top-K, projections, and image quality metrics
│   ├── models.py                      # ResNet-18 & WRN-28-10 backbones with checkpoint integrity checks
│   ├── training/                      # Clean & Adversarial training engines
│   └── utils.py                       # Seed, device, and hash reproducibility utilities
│
├── scripts/
│   ├── attack_benchmark.py            # CLI runner for comprehensive multi-baseline benchmark
│   ├── defense_benchmark.py           # CLI runner for defense evaluation (oblivious & adaptive BPDA)
│   ├── run_casa_benchmark.py          # High-performance runner for CASA SOTA benchmark
│   ├── run_ablation.py                # Component ablation study runner across 10 variants
│   ├── analyze_failures.py            # Diagnostic failure case analysis tool
│   ├── plot_paper_figures.py          # Publication-ready figure generator (Figures 1 to 7)
│   ├── kaggle_run.sh                  # Automated master execution script for Kaggle GPU
│   ├── evaluate_checkpoint.py         # Checkpoint SHA256 integrity and accuracy verification
│   └── generate_markdown_report.py    # Report generator from JSON benchmark results
│
├── tests/                             # PyTest unit, contract, and benchmark verification tests
├── result/                            # Verified raw JSON artifacts, manifests & figures
└── third_party/                       # Pinned upstream official author implementations
```

---

## 2. Research Documentation Roadmap

The project documentation is organized into **14 canonical documents** in [`docs/`](docs/):

| Document | Title & Scope | Key Focus |
| :--- | :--- | :--- |
| [**01 Research Overview**](docs/01_research_overview.md) | Research Overview & Scope | Problem statement, RQ1–RQ5, core contributions, Track A vs B. |
| [**02 Threat Model & Protocol**](docs/02_threat_model_and_protocol.md) | Single Source of Truth (SSOT) | Spatial $L_0$, $\tau=10^{-5}$, clean-correct denominator, ASR/CRA, $F+2B$. |
| [**03 CASA Method**](docs/03_casa_method.md) | Method Specification | DLR objective, Headroom Gain, Spatial NMS, 1-swap/2-swap, Drop-repair. |
| [**04 Datasets**](docs/04_datasets.md) | Datasets & Registry | CIFAR-10, CIFAR-100, Tiny-ImageNet, test indices hash, normalization. |
| [**05 Models & Training**](docs/05_models_and_training.md) | Target Models & Checkpoints | Model acceptance gate (7/7 pass), training protocol, SHA256 registry. |
| [**06 Baselines**](docs/06_baselines.md) | Baseline Methodology | SPGD, Sparse-RS, CornerSearch, Sigma-Zero, SparseFool, GSE, PGD0. |
| [**07 Experiment Plan**](docs/07_experiment_plan.md) | Operational Plan | P0–P5 execution phases, compute funnel, Go/No-Go decision gates. |
| [**08 Attack Benchmark**](docs/08_attack_benchmark.md) | Empirical Results Report | 10k CIFAR-10 benchmark tables, low-$K$ sparsity advantage, Pareto frontier. |
| [**09 Ablation & Analysis**](docs/09_ablation_and_analysis.md) | Ablation Studies & Failures | 10-variant breakdown, coalition synergy $I(i,j)$, sensitivity, counterfactuals. |
| [**10 Defense Research**](docs/10_defense_research.md) | Sparse Defense Track | Preprocessing filters, BPDA adaptive gap ($>+30\%$), gradient masking tests. |
| [**11 Reproducibility**](docs/11_reproducibility.md) | Reproduction Cookbook | 10-step reproduction guide, Kaggle GPU execution, multi-GPU handling. |
| [**12 Artifacts & Results**](docs/12_artifacts_and_results.md) | Artifacts & JSON Schemas | Paper table/figure mapping to raw JSON, `manifest.json` schema, SHA256 sums. |
| [**13 Limitations**](docs/13_limitations.md) | Boundary Conditions | Combinatorial limits, greedy optima, boundary saturation, negative scope. |
| [**14 Publication Checklist**](docs/14_publication_checklist.md) | Release Gate (DoD) | Quality gates, verification checklists, Publication Readiness status. |

---

## 3. Quick Start

### Step 1: Environment Setup
```bash
git clone https://github.com/nxc1802/AA_2.git
cd AA_2
pip install -e .
```

### Step 2: Verify Checkpoint Integrity (Mandatory Gate)
```bash
python scripts/evaluate_checkpoint.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint result/saved_models/resnet18_cifar10_best.pth \
    --expected-sha256 378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172 \
    --expected-acc 94.84
```

### Step 3: Run Sanity Smoke Test (< 1 min)
```bash
python scripts/attack_benchmark.py --config configs/smoke.yaml --strict
```

### Step 4: Run Intermediate Validation (1,000 samples)
```bash
python scripts/attack_benchmark.py --config configs/development.yaml
```

### Step 5: Run Official 10,000-Sample Paper Benchmark
```bash
# Run all 8 baselines + CASA on 10,000 CIFAR-10 test samples
python scripts/attack_benchmark.py --config configs/paper_cifar10.yaml

# Or run dedicated CASA evaluation across 7 budgets
python scripts/run_casa_benchmark.py --k 1 2 4 8 16 32 64 --batch-size 16 --samples 1000
```

### Step 6: Run Component Ablation Study
```bash
python scripts/run_ablation.py --samples 1000 --k-values 1 4 16
```

### Step 7: Run Adaptive Defense Benchmark (BPDA)
```bash
python scripts/defense_benchmark.py --config configs/development.yaml --defense median --mode adaptive
```

### Step 8: Run Unit & Contract Tests
```bash
pytest tests/ -v
```

---

## 4. Large-Scale Execution on Kaggle GPU

For full 10,000-sample execution across all baselines without local GPU resources:
- Use [`kaggle_paper_runner.ipynb`](kaggle_paper_runner.ipynb) directly on a Kaggle GPU instance (2× Tesla T4).
- Or run the automated script:
  ```bash
  bash scripts/kaggle_run.sh all
  ```
See [`docs/11_reproducibility.md`](docs/11_reproducibility.md) for detailed instructions.

---

## 5. Provenance & License

- Core codebase is licensed under the [MIT License](LICENSE).
- Upstream baseline provenance, pinned commit SHAs, and original licenses are cataloged in [THIRD_PARTY.md](THIRD_PARTY.md).
