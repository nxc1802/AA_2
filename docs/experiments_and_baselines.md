# Experimental Protocol, Scope & Competitor Baseline Guide

This document serves as the comprehensive single source of truth for the experimental scope, configuration architecture, third-party competitor baselines, and step-by-step execution workflows for the sparse adversarial attack benchmark.

---

## 1. Research Scope & Evaluation Protocol

### 1.1 Threat Model
We study pixel-sparse adversarial attacks under the **spatial $L_0$ pseudo-norm**. For a clean image $x \in [0, 1]^{3 \times H \times W}$ and adversarial image $x_{\text{adv}} \in [0, 1]^{3 \times H \times W}$, the perturbation is $\delta = x_{\text{adv}} - x$.

A spatial pixel coordinate $(h, w)$ is defined as perturbed if any of its color channels change by more than numerical threshold $\tau = 10^{-5}$:

$$\|\delta\|_{0, \text{spatial}} = \sum_{h=1}^H \sum_{w=1}^W \mathbb{I}\left( \max_{c \in \{1, 2, 3\}} |\delta_{c, h, w}| > \tau \right) \le K$$

Evaluation is conducted across a logarithmic sequence of sparse pixel budgets:
$$K \in \{1, 2, 4, 8, 16, 32, 64\}$$
For an image of size $32 \times 32$ (1,024 spatial pixels), $K=1$ corresponds to **$0.098\%$** sparsity, and $K=16$ corresponds to **$1.56\%$** sparsity.

### 1.2 Datasets & Splits
1. **Primary Benchmark: CIFAR-10**
   - 10 classes, $3 \times 32 \times 32$ RGB images.
   - Clean training split: 40,000 images; Validation split: 10,000 images (class-stratified, seed 42).
   - Test evaluation set:
     - **1,000-sample validation cohort**: Uniformly stratified across all 10 classes (100 samples/class), deterministic seed 42. Used for rigorous intermediate verification, rapid prototyping, and hyperparameter search.
     - **10,000-sample full test set**: Used for official final paper camera-ready tables.
2. **Extended Benchmark: CIFAR-100**
   - 100 classes, $3 \times 32 \times 32$ RGB images.
   - Evaluates whether sparse vulnerability scales with target semantic granularity.
3. **Large-Scale Scale-Up: ImageNet-1k**
   - 1,000 classes, $3 \times 224 \times 224$ images.
   - Evaluates whether spatial coalition mechanics remain computationally tractable as spatial resolution increases by $49 \times$.

### 1.3 Target Model Architectures
- **Primary Backbone: CIFAR-adapted ResNet-18**
  - Architecture: Initial $3 \times 3$ convolution with stride 1, no initial max-pooling layer, standard residual stage configuration `[2, 2, 2, 2]`.
  - Canonical Checkpoint: `result/saved_models/resnet18_cifar10_best.pth`.
  - Checksum: `SHA256: 378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172`.
  - Performance: **94.84%** accuracy on full test set, **93.70%** accuracy on 1,000-sample evaluation cohort.
- **Secondary Backbones (Generalization Benchmarks)**:
  - **WideResNet-28-10**: High-capacity wide convolutional network.
  - **ResNet-50**: Deep bottleneck architecture.
  - **Vision Transformer (ViT-Tiny / ViT-Small)**: Evaluates patch-based self-attention robustness against localized pixel attacks.

### 1.4 Standardized Evaluation Metrics
- **Conditional Attack Success Rate ($ASR@K$)**:
  $$\text{ASR}@K = \frac{\sum_{i=1}^N \mathbb{I}\left(f(x_i + \delta_i) \neq y_i \land f(x_i) = y_i \land \|\delta_i\|_{0, \text{spatial}} \le K\right)}{\sum_{i=1}^N \mathbb{I}\left(f(x_i) = y_i\right)} \times 100\%$$
  Samples classified incorrectly on clean images are strictly excluded from the ASR denominator to avoid artificial inflation.
- **Conditional Robust Accuracy ($CRA@K$)**: $100\% - \text{ASR}@K$.
- **Geometric Perturbation Norms**: Mean and median spatial $L_0$, channel $L_0$, $L_2$, and $L_\infty$.
- **Perceptual Image Quality**: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
- **Computational Efficiency**: Average queries per image, forward evaluations, backward evaluations, and wall-clock execution time.

---

## 2. Configuration System & File Architecture

The repository utilizes a modular YAML configuration hierarchy:

```
configs/
├── paper_cifar10.yaml           # Official multi-attack benchmark configuration
├── casa.yaml                    # Dedicated CASA SOTA evaluation configuration
├── smoke.yaml                   # Fast smoke test configuration for CI/CD
├── train/
│   ├── clean_cifar10_resnet18.yaml
│   └── clean_cifar10_wrn28_10.yaml
└── adv_train/
    ├── pgd_at_cifar10_resnet18.yaml
    └── spgd_at_cifar10_resnet18.yaml
```

### 2.1 Main Benchmark Configuration (`configs/paper_cifar10.yaml`)
```yaml
seed: 42
strict: true  # Abort immediately if any checkpoint checksum or accuracy check fails

dataset:
  name: cifar10
  samples: 1000  # Set to 10000 for full paper test set
  batch_size: 512

model:
  name: resnet18
  checkpoint: resnet18_cifar10_best.pth
  expected_sha256: 378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172
  expected_clean_acc: 94.84

benchmark:
  k_values: [1, 2, 4, 8, 16, 32, 64]

# Per-method batch sizes (optimized for GPU/MPS memory)
attacks_batch_size:
  fgsm: 512
  bim: 512
  pgd: 512
  cornersearch: 512
  pgd0: 512
  spgd: 512
  sparse_rs: 512
  sparsefool: 512
  sigma_zero: 512
  gse: 512
  casa: 16  # Coalition search optimal batch size
```

---

## 3. Third-Party Competitor Baselines

To ensure fair and rigorous comparison, all 10 competitor baselines are categorized into distinct paradigms.

```
+------------------------------------------------------------------------------------------------+
|                                     Baseline Taxonomies                                        |
+------------------------------------+-----------------------------+-----------------------------+
| Paradigm                           | Methods                     | Nature                      |
+------------------------------------+-----------------------------+-----------------------------+
| 1. Dense Reference Baselines       | FGSM, BIM, PGD              | L_inf bounded, L0 = 1024    |
| 2. Blackbox Sparse Competitors     | CornerSearch, Sparse-RS     | Derivative-free / Zeroth-ord|
| 3. Whitebox Sparse Competitors     | SPGD, Sigma-Zero,           | First-order gradient-based  |
|                                    | SparseFool, PGD0, GSE       | continuous / discrete       |
| 4. Proposed SOTA Whitebox          | CASA                        | Coalition Game Theory       |
+------------------------------------+-----------------------------+-----------------------------+
```

### 3.1 Dense Reference Baselines ($L_\infty$-bounded)
*Note: Dense attacks modify all 1,024 pixels and serve only as reference points for classification fragility.*
1. **FGSM (Fast Gradient Sign Method)**
   - **Math**: $x_{\text{adv}} = \Pi_{[0, 1]} \left( x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}_{\text{CE}}(x, y)) \right)$
   - **Config**: $\epsilon = 8/255 \approx 0.0314$, single-step forward-backward.
2. **BIM (Basic Iterative Method / I-FGSM)**
   - **Math**: Multi-step FGSM with clipping to $\epsilon$-ball: $x^{(t+1)} = \Pi_{x, \epsilon} \left( x^{(t)} + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}) \right)$
   - **Config**: $\epsilon = 8/255$, $\alpha = 2/255$, $\text{steps} = 10$.
3. **PGD (Projected Gradient Descent - $L_\infty$)**
   - **Math**: Multi-step iterative ascent with uniform random start within $[x - \epsilon, x + \epsilon]$.
   - **Config**: $\epsilon = 8/255$, $\alpha = 2/255$, $\text{steps} = 20$, $\text{random\_start} = \text{True}$.

### 3.2 Blackbox Sparse Competitors
4. **CornerSearch (Brendel et al., Croce et al.)**
   - **Paradigm**: Blackbox Greedy Search.
   - **Mechanism**: Evaluates the 8 discrete corners $\{0, 1\}^3$ of the RGB color cube on pixels ranked by 1-pixel modification impact. Iteratively selects the best single pixel until success or $K_{\max} = 64$.
   - **Implementation**: Evaluated in progressive mode where $ASR@K$ curves for all $K \in \{1, 2, \dots, 64\}$ are extracted from each sample's achieved $L_0$.
   - **Hyperparameters**: $K_{\max} = 64$.
   - **Characteristics**: Extremely high query consumption ($68,943$ queries/img, $\sim 4.8$ hours for 1,000 samples).
5. **Sparse-RS (Croce et al., ECCV 2020)**
   - **Paradigm**: Blackbox Random Search with geometric square candidates.
   - **Mechanism**: Randomly samples candidate pixel subsets, testing 3D RGB corner perturbations. Adjusts candidate set size adaptively based on target margin reduction.
   - **Hyperparameters**: $N_{\text{queries}} = 10,000$, $\alpha_{\text{init}} = 0.3$.
   - **Characteristics**: Achieves 100% ASR at $K \ge 16$ but requires $10,000$ queries/image ($>3.2$ hours runtime).

### 3.3 Whitebox Sparse Competitors
6. **SPGD (Sparse PGD - Croce & Hein, ICML 2019)**
   - **Paradigm**: Whitebox Continuous Relaxation & Simplex Projection.
   - **Mechanism**: Relaxes the combinatorial $L_0$ constraint into continuous optimization over a simplex, followed by projection onto the $K$-sparse boundary.
   - **Hyperparameters**: $\text{steps} = 100$, step size adaptively scheduled.
   - **Performance**: Former whitebox SOTA ($60.09\%$ at $K=4$, $86.23\%$ at $K=8$), but drops to $14.51\%$ at $K=1$ due to continuous gradient dispersion.
7. **Sigma-Zero (Minimal Perturbation Attack)**
   - **Paradigm**: Whitebox Sigmoid-Relaxed Minimal $L_0$ Optimization.
   - **Mechanism**: Optimizes a continuous mask $m \in [0, 1]^{H \times W}$ modulated by a temperature parameter $\sigma \to 0$ to encourage binary sparsity.
   - **Hyperparameters**: $\text{steps} = 500$, loss weight $\lambda = 0.01$.
8. **SparseFool (Modas et al., CVPR 2019)**
   - **Paradigm**: DeepFool Extension for $L_1/L_0$ boundaries.
   - **Mechanism**: Computes first-order affine approximations of classification decision boundaries, project onto $L_1$ ball, and aggregates coordinates to minimize $L_0$.
   - **Hyperparameters**: $\text{max\_iter} = 20$, $\lambda = 3.0$.
9. **PGD0 (Greedy Saliency Masking PGD)**
   - **Paradigm**: Fixed-Budget Gradient Masking.
   - **Mechanism**: Computes top-$K$ gradient locations on clean image and executes 100 PGD iterations restricted to that static mask.
   - **Hyperparameters**: $\text{steps} = 100$, $\alpha = 0.1$.
   - **Limitation**: Static pixel selection fails when gradient directions change dynamically during ascent ($ASR = 8.22\%$ at $K=4$).
10. **GSE (Greedy Saliency Estimation)**
    - **Paradigm**: Whitebox Saliency-Based Greedy Pursuit.
    - **Mechanism**: Computes saliency maps and substitutes pixels iteratively one by one.
    - **Hyperparameters**: $\text{max\_evals} = 1000$.

---

## 4. Step-by-Step Execution Guide

All execution commands must be launched from the workspace root (`/Volumes/WorkSpace/Project/AA`).

### Step 4.1: Clean Model Verification & Checksum Gate
Before running attacks, verify that the clean checkpoint exists and satisfies the SHA256 security gate:
```bash
python scripts/evaluate_checkpoint.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint result/saved_models/resnet18_cifar10_best.pth \
    --expected-sha256 378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172
```

### Step 4.2: Running the Competitor Baseline Suite
To execute all third-party baselines on the 1,000-sample CIFAR-10 cohort:
```bash
python scripts/attack_benchmark.py \
    --config configs/paper_cifar10.yaml \
    --output result/baseline_benchmark_results.json
```

To run a specific competitor attack (e.g., SPGD or CornerSearch):
```bash
python scripts/attack_benchmark.py \
    --config configs/paper_cifar10.yaml \
    --attacks spgd \
    --k 1 2 4 8 16 32 64
```

### Step 4.3: Running the Proposed SOTA Method (CASA)
To execute the official CASA SOTA attack benchmark:
```bash
python scripts/run_casa_benchmark.py \
    --k 1 2 4 8 16 32 64 \
    --batch-size 16 \
    --samples 1000 \
    --device mps \
    --output result/casa_sota_results.json
```

### Step 4.4: Running the Defense Benchmark
To evaluate adversarial attacks against preprocessing defenses (Median, TVM, JPEG, Gaussian Blur) under both oblivious and adaptive (BPDA) settings:
```bash
python scripts/defense_benchmark.py \
    --config configs/paper_cifar10.yaml \
    --defense median \
    --mode adaptive \
    --output result/defense_median_adaptive.json
```

### Step 4.5: Generating Comparative Markdown Reports
To synthesize raw JSON benchmark results into publishable Markdown tables:
```bash
python scripts/generate_markdown_report.py \
    --results result/baseline_benchmark_results.json result/casa_sota_results.json \
    --output docs/benchmark_results_cifar10.md
```

---

## 5. Scaling to Extended Scenarios (ResNet-50, CIFAR-100, ImageNet)

### 5.1 Training Clean Backbones for New Scenarios
To train a clean ResNet-50 or WideResNet-28-10 model on CIFAR-100:
```bash
python scripts/train_clean.py \
    --config configs/train/clean_cifar100_resnet50.yaml \
    --epochs 200 \
    --device cuda
```

### 5.2 Adapting Batch Sizes for Deeper Backbones
Because CASA evaluates coalitions via backpropagation, GPU memory scales with architecture depth and spatial resolution. Recommended batch sizes:

| Architecture | Dataset | Resolution | Recommended CASA Batch Size |
| :--- | :--- | :---: | :---: |
| **ResNet-18** | CIFAR-10 / 100 | $32 \times 32$ | **16 – 32** |
| **WideResNet-28-10** | CIFAR-10 / 100 | $32 \times 32$ | **8 – 16** |
| **ResNet-50** | CIFAR-100 | $32 \times 32$ | **8 – 16** |
| **ResNet-50** | ImageNet-1k | $224 \times 224$ | **2 – 4** |
| **ViT-Small** | CIFAR-10 / ImageNet | $224 \times 224$ | **2 – 4** |
