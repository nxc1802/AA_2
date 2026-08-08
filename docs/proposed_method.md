# Proposed Method: Sparse Feature Attack (SFA)

> Status: Active Implementation (`src/aa/attacks/proposed.py`)

The proposed method combines feature-space disruption, spatial support scoring, sparse $L_0$ optimization, and iterative support pruning.

---

## 1. Problem Formulation

Given a classifier $f$, clean image $x$, and true label $y$, the objective is:

$$\min_{\delta} \|\delta\|_{0,\mathrm{spatial}} \quad \text{s.t.} \quad f(x+\delta) \ne y \quad \text{and} \quad x+\delta \in [0, 1]^d.$$

Rather than selecting pixels solely based on output gradient magnitudes, the method evaluates feature representation disruption in combination with local spatial support interaction.

---

## 2. Pipeline Overview

```text
clean image
    │
    ▼
critical feature extraction
    │
    ├──────────────┐
    ▼              ▼
output gradient   feature gradient
    │              │
    └──────┬───────┘
           ▼
    support scoring
           │
           ▼
       top-K support
           │
           ▼
   sparse optimization
           │
           ▼
 successful candidate
           │
           ▼
     support pruning
           │
           ▼
minimal / near-minimal
 adversarial support
```

---

## 3. Loss & Support Selection

Let $\phi(x)$ denote intermediate layer features (e.g., `layer4` of ResNet-18).

The joint loss is:

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}}(f(x_{\mathrm{adv}}), y) + \lambda_f \|\phi(x_{\mathrm{adv}}) - \phi(x)\|_2^2.$$

Spatial support mask selection:

$$S_K = \operatorname{TopK}(\text{Score}(x), K).$$

Update step:

$$x_{t+1} = \Pi_{[0,1]} \left( x_t + \alpha M_{S_K} \odot \operatorname{sign}(\nabla_x \mathcal{L}(x_t, y)) \right).$$

Followed by exact $L_0$ projection:

$$\delta_{t+1} = \operatorname{Project}_{L_0}(\delta_{t+1}, K).$$

---

## 4. Support Pruning

Upon finding a successful candidate adversarial perturbation $x_{\mathrm{adv}}$:

1. Identify active modified spatial locations.
2. Iteratively attempt setting individual active pixel modifications to 0.
3. Re-evaluate prediction on pruned candidate.
4. Keep the modification zeroed out if misclassification is preserved.
5. Produce final near-minimal support perturbation.

---

## 5. Ablation Components

The implementation supports clean component ablation via parameter flags:

* **A0 (Base)**: `feature_guidance=False, interaction=False, pruning=False`
* **A1 (Feature)**: `feature_guidance=True, interaction=False, pruning=False`
* **A2 (Interaction)**: `feature_guidance=True, interaction=True, pruning=False`
* **A3 (Full Method)**: `feature_guidance=True, interaction=True, pruning=True`