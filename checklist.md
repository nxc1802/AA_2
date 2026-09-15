Có. Mình đề xuất coi đây là **checklist “CASA → paper-ready / public-ready”**, ưu tiên theo P0/P1/P2. Mục tiêu là không thêm heuristic tùy tiện, mà **khóa protocol và chứng minh claim**.

## 🔴 P0 — Bắt buộc trước khi claim SOTA

### 1. Khóa benchmark protocol

* [ ] Chốt threat model: targeted hay untargeted.
* [ ] Chốt spatial \(L_0\), threshold \(\tau\).
* [ ] Chốt \(K=\{1,2,4,8,16,32,64\}\).
* [ ] Chốt clean model/checkpoint + SHA256.
* [ ] `paper_cifar10.yaml` mặc định **10,000 test images**, không phải 1,000.
* [ ] Seed = 42 cho official benchmark.
* [ ] Không tune hyperparameter trên test set.
* [ ] Tách rõ:

  * [ ] fixed-budget attacks
  * [ ] minimum-support attacks.
* [ ] Viết protocol thành một document duy nhất.

**Done khi:** một người khác clone repo và chạy đúng một command → reproduce official table.

---

### 2. Benchmark lại toàn bộ baseline ở setting mạnh nhất

Đây là **P0 quan trọng nhất**.

* [ ] Sparse-PGD official implementation.

  * [ ] projected
  * [ ] unprojected nếu applicable
  * [ ] official/recommended iteration budget
* [ ] Sigma-Zero

  * [ ] official implementation
  * [ ] official iteration budget
* [ ] PGD-L0
* [ ] Sparse-RS
* [ ] SparseFool
* [ ] GSE
* [ ] SAIF nếu phù hợp threat model
* [ ] Sparse-AutoAttack / sAA
* [ ] CornerSearch
* [ ] PGD0

Đặc biệt:

> **Không được chỉ benchmark SPGD = 100 steps rồi kết luận CASA > SPGD.**

Phải benchmark với **strongest reasonable configuration**.

---

### 3. Full CIFAR-10 10k

Chạy:

* [ ] CASA — 10k
* [ ] SPGD — 10k
* [ ] Sigma-Zero — 10k
* [ ] Sparse-RS — 10k
* [ ] PGD-L0 — 10k
* [ ] SparseFool — 10k
* [ ] GSE — 10k
* [ ] sAA — 10k

Cho:

```text
K = 1, 2, 4, 8, 16, 32, 64
```

Lưu raw result, không chỉ Markdown.

---

### 4. Statistical significance

Cho K=1/2/4 ít nhất:

* [ ] ASR
* [ ] 95% CI
* [ ] absolute improvement
* [ ] relative improvement

Nếu compute cho phép:

* [ ] 3 independent seeds.

Main claim phải dựa trên **10k**, không phải 1k.

---

# 🔴 P0 — Ablation để chứng minh CASA thực sự có contribution

Hiện CASA có khá nhiều component. Cần chứng minh component nào tạo improvement.

Chạy:

| Variant                     | K=1 | K=4 | K=16 |
| --------------------------- | --: | --: | ---: |
| Base sparse gradient attack |     |     |      |
| + Box-aware gain            |     |     |      |
| + GCT                       |     |     |      |
| + Spatial NMS               |     |     |      |
| + Support exchange          |     |     |      |
| + Pair exploration          |     |     |      |
| + Dynamic refresh           |     |     |      |
| + Tabu                      |     |     |      |
| + Drop-and-repair           |     |     |      |
| **Full CASA**               |     |     |      |

### Cần đặc biệt trả lời:

* [ ] GCT đóng góp bao nhiêu?
* [ ] NMS đóng góp bao nhiêu?
* [ ] coalition/support exchange đóng góp bao nhiêu?
* [ ] pair swap có thực sự cần thiết?
* [ ] drop-and-repair cải thiện actual \(L_0\) bao nhiêu?
* [ ] heuristic nào chỉ tăng runtime nhưng không tăng ASR?

**Nếu một component không đóng góp → bỏ.**

Đừng giữ 12 heuristic chỉ vì “có vẻ tốt”.

---

# 🔴 P0 — Làm rõ novelty / mathematical formulation

Đây là phần mình khuyên sửa mạnh.

### Core formulation

Viết CASA thành:

$$
\min_{S:|S|\le K}
\min_{\delta:\operatorname{supp}(\delta)\subseteq S}
L(x+\delta,y)
$$

hoặc maximization equivalent.

Sau đó:

$$
S_t
\rightarrow
\delta_{S_t}
\rightarrow
S_{t+1}
\rightarrow
\delta_{S_{t+1}}
$$

### Cần định nghĩa rõ:

* [ ] Support value \(F(S)\)
* [ ] Candidate gain
* [ ] Box-aware gradient score
* [ ] Support exchange
* [ ] Pair swap
* [ ] Drop-and-repair.

### Quan trọng:

Nếu implementation không thực sự tính:

$$
F(S\cup\{j\})-F(S)
$$

thì **đừng overclaim “game-theoretic marginal contribution”.**

Dùng terminology:

> **coalition-aware gradient-guided support search**

sẽ an toàn hơn.

---

# 🔴 P0 — Efficiency benchmark

Không chỉ báo `queries/image`.

Tạo bảng:

| Method     | ASR | Forward | Backward | Grad eval | Time/img | Actual L0 |
| ---------- | --: | ------: | -------: | --------: | -------: | --------: |
| SPGD       |     |         |          |           |          |           |
| Sigma-Zero |     |         |          |           |          |           |
| CASA       |     |         |          |           |          |           |

* [ ] Định nghĩa chính xác “query”.
* [ ] Tách forward/backward.
* [ ] Report wall-clock.
* [ ] Report GPU.
* [ ] Report batch size.
* [ ] Report memory nếu có thể.
* [ ] Không so sánh black-box query với white-box gradient cost như cùng một đơn vị.

---

# 🟠 P1 — Generalization

Sau khi CIFAR-10 đã chắc chắn:

### Model

* [ ] ResNet-18
* [ ] WRN-28-10

### Dataset

* [ ] CIFAR-10
* [ ] CIFAR-100

Nếu kết quả tốt:

* [ ] ImageNet
* [ ] ResNet-50

**Không cần làm tất cả ngay.**

CIFAR-10 + WRN-28-10 đã quan trọng hơn việc có 5 architecture nhưng benchmark chưa chắc.

---

# 🟠 P1 — Robustness của claim

Chạy thêm:

### Different seeds

* [ ] seed 0
* [ ] seed 1
* [ ] seed 2

### Different initialization

* [ ] gradient initialization
* [ ] random initialization
* [ ] GCT initialization

### Hyperparameter sensitivity

Đối với:

```text
steps
inner_steps
alpha
repair_steps
pair_search_every
```

* [ ] ±20–30% parameter variation.
* [ ] chứng minh CASA không chỉ thắng nhờ một hyperparameter cherry-pick.

---

# 🟠 P1 — Actual L0 analysis

Đây có thể trở thành **contribution figure rất mạnh**.

Report:

$$
ASR \quad vs \quad K
$$

và:

$$
ASR \quad vs \quad \text{actual }L_0
$$

Cụ thể:

* [ ] mean L0
* [ ] median L0
* [ ] percentile 25/50/75
* [ ] ASR theo actual L0.

Đặc biệt cần highlight:

> CASA đạt ASR cao với support nhỏ đến mức nào?

---

# 🟠 P1 — Failure analysis

Đừng chỉ show success.

Lấy:

* [ ] 20–50 CASA failures tại K=1
* [ ] 20–50 failures tại K=2
* [ ] 20–50 failures tại K=4

Phân tích:

* [ ] object/background?
* [ ] class confusion?
* [ ] gradient issue?
* [ ] saturation?
* [ ] local minima?
* [ ] support interaction?

Và:

> CASA fail ở những trường hợp nào mà SPGD thành công?

Đây có thể dẫn tới insight rất tốt.

---

# 🟠 P1 — Visualization

Tạo figure chuẩn paper:

### Figure 1

Clean → adversarial

show:

```text
K=1
K=2
K=4
K=8
```

### Figure 2

ASR vs K.

### Figure 3

Actual L0 vs K.

### Figure 4

ASR vs computation.

### Figure 5

Ablation.

### Figure 6

Visualization support evolution:

```text
S0 → S1 → S2 → ... → SK
```

Cái này rất hợp với CASA.

---

# 🟠 P1 — Defense evaluation

Giữ defense nhưng **không dùng nó làm bằng chứng SOTA**.

* [ ] Median
* [ ] Gaussian blur
* [ ] JPEG
* [ ] TVM
* [ ] BPDA adaptive attack
* [ ] CASA-AT nếu muốn.

Quan trọng:

> defense phải được đánh giá với **adaptive CASA**, không chỉ oblivious attack.

---

# 🟡 P2 — Code cleanup trước public

### Naming

Hiện có:

```text
ours
casa
ours_v1
```

Nên chuẩn hóa thành:

```text
casa
casa_v1
casa_v2
```

hoặc chỉ:

```text
casa
```

---

### Config

Tạo:

```text
configs/
├── smoke.yaml
├── development.yaml
└── paper_cifar10.yaml
```

Trong đó:

```text
smoke       → 20 samples
development → 1,000
paper       → 10,000
```

`paper_cifar10.yaml` **không được yêu cầu user sửa sample count bằng tay**.

---

### Reproducibility

* [ ] lock dependency versions.
* [ ] record Python version.
* [ ] PyTorch version.
* [ ] CUDA version.
* [ ] GPU model.
* [ ] OS.
* [ ] upstream commit SHA.
* [ ] checkpoint SHA256.
* [ ] random seed.
* [ ] config hash.
* [ ] git commit hash.

---

# 🟡 P2 — Paper-facing benchmark table

Mình khuyên cuối cùng chỉ có một bảng chính kiểu:

| Attack     | Type   | K=1 | K=2 | K=4 | K=8 | K=16 | Queries/Grad | Time |
| ---------- | ------ | --: | --: | --: | --: | ---: | -----------: | ---: |
| SPGD       | WB     |     |     |     |     |      |              |      |
| Sigma-Zero | WB     |     |     |     |     |      |              |      |
| PGD-L0     | WB     |     |     |     |     |      |              |      |
| Sparse-RS  | BB     |     |     |     |     |      |              |      |
| CASA       | **WB** |  ** |  ** |  ** |     |      |              |      |

Sau đó **highlight chỉ những column CASA thực sự thắng**.

Đừng tô đậm toàn bộ CASA nếu không thắng toàn bộ.

---

# 🟢 P2 — Public release

Trước khi public:

* [ ] README không còn chữ “SOTA” nếu chưa verify.
* [ ] README có exact reproduction command.
* [ ] LICENSE.
* [ ] Citation / BibTeX.
* [ ] Dataset instructions.
* [ ] pretrained checkpoint instructions.
* [ ] baseline provenance.
* [ ] known limitations.
* [ ] expected runtime.
* [ ] expected GPU memory.
* [ ] raw benchmark JSON.
* [ ] generated tables.
* [ ] test suite pass.
* [ ] clean checkpoint verification pass.

---

# 🏁 Definition of Done

Mình sẽ coi CASA **paper-ready** khi thỏa 8 điều này:

### Scientific

* [ ] **10k CIFAR-10**
* [ ] **strongest current baselines**
* [ ] **sPGD properly tuned**
* [ ] **Sigma-Zero properly tuned**
* [ ] **sAA evaluated**
* [ ] **3-seed / confidence interval**
* [ ] **full ablation**
* [ ] **actual L0 + efficiency analysis**

### Claim

Sau khi hoàn thành, nếu kết quả kiểu:

```text
K=1    CASA +8~10%
K=2    CASA +6~8%
K=4    CASA +3~5%
K>=8   CASA ≈ strongest baseline
```

thì **không cần ép claim “overall SOTA”**.

Claim đẹp hơn:

> **CASA establishes a new state of the art for ultra-sparse white-box adversarial attacks at K≤4, while achieving competitive performance at larger budgets.**

Đây là claim **mạnh nhưng defendable**.

---

# ⭐ Thứ tự thực hiện mình khuyên

Đừng làm theo thứ tự file trong repo. Làm đúng thứ tự này:

```text
1. LOCK PROTOCOL
       ↓
2. STRONG BASELINES
       ↓
3. 10K CIFAR-10
       ↓
4. CASA 10K
       ↓
5. STATISTICAL TEST
       ↓
6. ABLATION
       ↓
7. EFFICIENCY
       ↓
8. WRN-28-10
       ↓
9. CIFAR-100
       ↓
10. IMAGE NET (optional)
       ↓
11. CLEAN UP THEORY
       ↓
12. CLEAN UP REPO
       ↓
13. REMOVE/REFINE SOTA CLAIM
       ↓
14. PAPER
       ↓
15. PUBLIC RELEASE
```

**Quan trọng nhất:** hiện tại mình **không khuyên tiếp tục phát triển CASA bằng cách thêm feature/heuristic**. Commit mới đã có rất nhiều cơ chế. Bước tiếp theo phải là **freeze algorithm → benchmark thật mạnh → ablation → chứng minh**. Nếu CASA vượt qua vòng này, lúc đó mới đáng đầu tư vào paper polishing và ImageNet.
