# Experimental Protocol

This document defines the experimental protocol used by AA_2.

The purpose of this file is to provide a single source of truth for model training, adversarial attack evaluation, defense evaluation, metrics, baseline selection, and reproducibility.

Implementation details may change during development, but paper-facing experiments must follow this protocol unless explicitly documented.

---

## 1. Research Scope

AA_2 studies pixel-sparse adversarial attacks under the spatial ($L_0$) threat model.

The primary research questions are:

1. How effective are modern sparse adversarial attacks under the same spatial ($L_0$) budget?
2. Can the proposed method achieve higher attack success with fewer modified pixels?
3. How does sparse robustness change under preprocessing and adversarial-training defenses?
4. Which components of the proposed method are responsible for its performance?

The main benchmark focuses on a small set of validated and reproducible baselines rather than maximizing the number of implemented attacks.

---

## 2. Datasets

### Primary dataset

CIFAR-10 is the primary benchmark.

* Training images: 50,000
* Test images: 10,000
* Image size: $3 \times 32 \times 32$
* Number of classes: 10

The official training split is divided into:

* 40,000 training images
* 10,000 validation images

The split must be deterministic, stratified by class, and generated using a fixed random seed (default 42).

The official 10,000-image test set is never used for model selection.

### Extended dataset

CIFAR-100 may be used as an additional generalization benchmark after the CIFAR-10 pipeline is stable.

The same train/validation/test protocol applies.

---

## 3. Models

The primary clean backbone is:

* CIFAR-adapted ResNet-18

The ResNet-18 architecture uses:

* $3\times3$ first convolution
* stride 1
* no initial max-pooling
* standard residual configuration ([2,2,2,2])
* dataset-specific final classifier

A secondary backbone may be used for generalization:

* WideResNet-28-10

Model-specific attack results must never be mixed across checkpoints.

---

## 4. Clean Training

Each model-dataset pair is trained once and the resulting checkpoint is reused for all attack experiments.

Default clean-training configuration:

```yaml
optimizer: SGD
learning_rate: 0.1
momentum: 0.9
weight_decay: 5e-4
epochs: 200
scheduler: cosine
train_batch_size: 256
eval_batch_size: 512
```

Training augmentation:

* RandomCrop(32, padding=4)
* RandomHorizontalFlip

Validation, test, and adversarial evaluation use deterministic preprocessing only.

The checkpoint with the best validation accuracy is saved.

At minimum, every released checkpoint must record:

```text
dataset
architecture
training seed
training configuration
validation accuracy
test accuracy
checkpoint SHA256
git commit
```

---

## 5. Threat Model

For clean image $x$ and adversarial image $x_{\mathrm{adv}}$,

$$\delta = x_{\mathrm{adv}} - x.$$

A spatial position is considered modified if at least one channel changes by more than numerical tolerance $\tau = 10^{-5}$:

$$\|\delta\|_{0,\mathrm{spatial}} = \sum_{h,w} \mathbf{1} \left[ \max_c |\delta_{c,h,w}| > \tau \right].$$

The main sparse-budget experiments use:

$$K \in \{1, 2, 4, 8, 16, 32, 64\}.$$

For a budgeted sparse attack,

$$\|\delta\|_{0,\mathrm{spatial}} \le K.$$

All images and perturbation metrics are evaluated in the pixel domain $[0, 1]$.

---

## 6. Attack Success

For untargeted attacks, an attack is successful when:

1. the clean model classifies the original image correctly, and
2. the adversarial prediction differs from the true label.

Conditional Attack Success Rate is defined as:

$$\mathrm{ASR} = \frac{\sum_i \mathbf{1}[f(x_i)=y_i] \mathbf{1}[f(x_i^{\mathrm{adv}})\ne y_i]}{\sum_i \mathbf{1}[f(x_i)=y_i]}.$$

Clean-incorrect samples are excluded from the ASR denominator but remain part of full-set robust accuracy reporting.

---

## 7. Main Baselines

### Dense references
* FGSM
* BIM
* PGD

### Budgeted sparse attacks
* CornerSearch
* PGD0
* Sparse-PGD
* Sparse-RS

### Minimal-support attacks
* SparseFool
* Sigma-Zero
* GSE

### Proposed method
* Ours (`SparseFeatureAttack`)

---

## 8. Baseline Validation Policy

A literature baseline is eligible for the main benchmark only when at least one of the following conditions holds:

1. a verified official-author implementation is used;
2. the method is a canonical reference algorithm with a simple independently testable implementation (FGSM, BIM, PGD);
3. a custom reimplementation has reproduced trusted reference behavior within documented tolerance.

Custom duplicate implementations are not used as canonical paper baselines when an official implementation exists.

---

## 9. Evaluation Samples

All attacks compared within the same experiment must use exactly the same sample indices.

Sampling must be deterministic, class-stratified, and controlled by a fixed seed.

---

## 10. Metrics

Report:
* Clean Accuracy
* Robust Accuracy
* Conditional ASR ($ASR@K$)
* Spatial $L_0$, relative sparsity $\rho = \frac{L_0}{HW}$
* Perturbation norms ($L_2$, $L_\infty$)
* Image quality (PSNR, SSIM, LPIPS)
* Wall-clock runtime, forward evaluations, backward evaluations, queries.
