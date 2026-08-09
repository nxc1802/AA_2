Mình đã đọc cấu trúc repo, protocol, proposed method, benchmark engine, metrics, model/data loading, defense pipeline, registry, một số adapter baseline, test suite, config paper và provenance của third-party code. Repo hiện là public Python project, default branch `main`, được tạo ngày 06/08/2026 và vừa cập nhật ngày 09/08/2026.

**Kết luận tổng quát:** kiến trúc project tốt, định hướng nghiên cứu rõ và tốt hơn khá nhiều so với một repo research kiểu notebook/script rời rạc. Tuy nhiên, ở trạng thái hiện tại mình **chưa xem đây là paper-ready benchmark**. Có vài vấn đề methodology và implementation đủ nghiêm trọng để làm sai hoặc làm yếu claim nếu chạy thẳng `configs/paper.yaml`.

## 1. Đánh giá nhanh

| Hạng mục                       |    Đánh giá |
| ------------------------------ | ----------: |
| Cấu trúc source code           |  **8.5/10** |
| Tách abstraction attack        |    **8/10** |
| Reproducibility design         |    **8/10** |
| Threat-model definition        |    **9/10** |
| Baseline coverage              |  **8.5/10** |
| Proposed-method implementation |  **6.5/10** |
| Benchmark fairness             |  **5.5/10** |
| Query/evaluation accounting    |    **4/10** |
| Defense evaluation             |    **5/10** |
| Adversarial training           |    **1/10** |
| Statistical rigor              |    **3/10** |
| Unit/integration testing       |    **5/10** |
| Paper readiness                | **~5.5/10** |

Điểm đáng khen nhất là repo đã có một **single source of truth cho experimental protocol** và định nghĩa đúng spatial (L_0), conditional ASR, common sample indices, separation giữa budgeted attacks và minimal-support attacks.

Nhưng implementation vẫn đang ở pha **research infrastructure + prototype proposed attack**, chưa phải benchmark artifact hoàn thiện.

---

# 2. Kiến trúc project

Repo được chia như sau:

```text
configs/
    paper.yaml

docs/
    protocol.md
    proposed_method.md
    roadmap.md
    upgrade.md

scripts/
    attack_benchmark.py
    defense_benchmark.py

src/aa/
    attacks/
        base.py
        dense.py
        proposed.py
        registry.py
        external/
            CornerSearch
            PGD0
            Sparse-PGD
            Sparse-RS
            SparseFool
            Sigma-Zero
            GSE
    benchmark.py
    data.py
    defenses.py
    metrics.py
    models.py
    utils.py

tests/

third_party/
```

README cũng mô tả đúng triết lý này: benchmark chung, attack adapters, exact spatial (L_0), preprocessing defenses, checkpoint loading và reproducibility utilities.

### Nhận xét

Đây là thiết kế đúng cho một research repo.

Thay vì:

```python
run_pgd.py
run_sparse_rs.py
run_ours.py
...
```

bạn có:

```python
Attack
  ↑
FGSM
PGD
SparseRS
PGD0
SFA
...
```

và một evaluator chung.

Điều này rất quan trọng vì giảm nguy cơ mỗi attack được tính metric theo một kiểu khác.

---

# 3. Experimental protocol — phần mạnh nhất

Protocol xác định rất rõ:

* CIFAR-10 primary
* 40k train / 10k validation
* test 10k
* ResNet-18 primary
* WRN-28-10 secondary
* (K={1,2,4,8,16,32,64})
* spatial (L_0)
* clean-correct conditional ASR
* common sample indices
* distortion metrics
* runtime / forward / backward / queries

Đặc biệt định nghĩa:

[
|\delta|_{0,\text{spatial}}
===========================

\sum_{h,w}
\mathbf1
\left[
\max_c |\delta_{c,h,w}| > 10^{-5}
\right]
]

là lựa chọn hợp lý hơn việc coi từng RGB channel là một dimension độc lập.

Code cũng implement đúng tinh thần đó:

```python
channel_max = delta.abs().max(dim=1).values
l0 = (channel_max > eps).flatten(1).sum(dim=1)
```

Đây là một điểm tốt để giữ nguyên trong paper.

---

# 4. Baseline selection tốt

Registry có:

### Dense references

* FGSM
* BIM
* PGD

### Sparse budget attacks

* CornerSearch
* PGD0
* Sparse-PGD
* Sparse-RS

### Minimal-support attacks

* SparseFool
* Sigma-Zero
* GSE

### Proposed

* SFA / `ours`

Bộ baseline này hợp lý.

Đặc biệt việc phân loại:

```python
mode = dense
mode = budget
mode = minimal
```

rất đúng về methodology.

Bạn không nên ép Sigma-Zero/SparseFool phải chạy riêng từng (K). Với minimal attack, chạy một lần rồi xây:

[
ASR@K
=====

\frac{
|{i:\ success_i \land L_0(i)\le K}|
}{
|{i:\ clean\ correct}|
}
]

là hợp lý. Benchmark đã có `derive_minimal_asr_curve()` cho việc này.

---

# 5. Proposed method SFA

Proposed method hiện có bốn thành phần chính:

[
\boxed{
\text{Feature guidance}
+
\text{gradient support score}
+
\text{interaction}
+
\text{support pruning}
}
]

Tài liệu mô tả objective:

[
\min_\delta |\delta|_{0,\text{spatial}}
]

subject to

[
f(x+\delta)\neq y.
]

Joint loss:

[
\mathcal L
==========

\mathcal L_{CE}
+
\lambda_f
|\phi(x_{adv})-\phi(x)|_2^2.
]

Implementation tương ứng khá sát với tài liệu.

## Feature guidance

Bạn hook vào:

* `layer4`
* `block3`
* `features`

và lấy feature difference.

Điểm tốt là nếu `feature_guidance=True` mà model không có layer phù hợp, code **fail loudly** thay vì silently bỏ feature loss.

Đây là design tốt.

---

# 6. Interaction mechanisms

Hiện code có 4 lựa chọn:

```text
cpa
fcsa
hsa
smoothing
```

### CPA

Dựa trên directional alignment giữa gradient pixel và 8 neighbors:

[
I_i
===

|g_i|
+
\lambda
\sum_{j\in N(i)}
\operatorname{ReLU}
(\cos(g_i,g_j))
|g_j|.
]

Cách implement tránh `torch.roll` và dùng padding là đúng, vì `roll` sẽ tạo wrap-around artifact ở biên.

Đây là chi tiết implementation tốt.

---

# 7. Nhưng novelty của interaction hiện chưa đủ mạnh

Đây là vấn đề quan trọng về paper.

Các tên:

* CPA — Cooperative Pixel Alignment
* FCSA — Functional Coalition Synergy
* HSA — Hypergraph...

nghe khá mạnh.

Nhưng implementation hiện tại phần lớn vẫn là **heuristic local gradient aggregation**.

Ví dụ FCSA:

```python
patch_max = max_pool(...)
patch_mean = avg_pool(...)
synergy = relu(patch_max - patch_mean)
```

HSA cũng chủ yếu là convolutions trên gradient magnitude ở 3×3, 5×5, 7×7.

Do đó mình sẽ rất cẩn thận với claim kiểu:

> “hypergraph cooperative reasoning”

hoặc

> “functional coalition modeling”

Nếu reviewer đọc implementation, họ có thể phản biện rằng đó chỉ là:

> multi-scale spatial smoothing / local gradient weighting.

### Gợi ý

Đổi framing thành thứ gần implementation hơn:

**Multi-scale spatial cooperation scoring**

hoặc

**Neighborhood-aware support ranking**

sẽ chắc chắn hơn.

CPA hiện là thành phần dễ defend nhất về mặt logic.

---

# 8. Critical issue #1 — SFA dừng ngay khi tìm thấy success

Trong loop:

```python
fooled_mask = fooled_mask | cand_succ
```

sau đó:

```python
score_masked[fooled_mask] = -inf
```

và update chỉ cho:

```python
(~fooled_mask)
```

Điều này nghĩa là một sample **ngừng optimization ngay sau lần misclassification đầu tiên**.

Sau đó mới chạy greedy pruning.

### Vấn đề

Nếu mục tiêu paper là:

> đạt ASR cao với support nhỏ nhất,

thì success đầu tiên chưa chắc là candidate tốt nhất.

Ví dụ:

```text
iteration 8  -> success at L0 = 16
iteration 15 -> có thể tìm support khác L0 = 11
iteration 22 -> có thể tìm support L0 = 8
```

Nhưng sample đã frozen từ iteration 8.

Greedy deletion chỉ tối ưu **trên support đã tìm thấy**, không explore support khác.

### Nên đổi

Có hai mode:

```text
fixed-budget attack
minimal-support attack
```

Với fixed budget, success có thể freeze nếu chỉ đo ASR@K.

Nhưng với proposed claim minimal/near-minimal, nên có:

1. continuation after success;
2. support shrink schedule;
3. hoặc binary search / progressive (K).

Ví dụ:

[
K_0=64
\rightarrow
32
\rightarrow
16
\rightarrow
8
...
]

và warm-start successful perturbation.

---

# 9. Critical issue #2 — support selection được recompute hoàn toàn mỗi step

Hiện:

```python
support_mask = topk(score, K)
candidate_delta =
    delta
    + alpha * grad.sign() * support_mask

delta = project_l0(candidate_delta, K)
```

Có một interesting dynamic ở đây:

* gradient Top-K chọn support mới;
* candidate giữ perturbation cũ;
* rồi `project_l0()` lấy Top-K theo perturbation magnitude.

Tức là có **hai ranking criteria liên tiếp**:

1. gradient interaction score;
2. accumulated perturbation (L_2)-magnitude.

Đây không hẳn sai, nhưng paper hiện không giải thích.

Và nó có thể gây:

```text
new high-gradient pixel
         ↓
small perturbation magnitude
         ↓
immediately discarded by project_l0
```

Nghĩa là support replacement có thể diễn ra chậm hoặc không diễn ra.

### Nên thiết kế rõ

Một trong:

[
S_{t+1}
=======

TopK(
(1-\beta)|\delta_t|
+
\beta Score_t
)
]

hoặc explicit:

```text
keep q existing support pixels
replace K-q positions
with highest-scoring candidates
```

Điều này sẽ biến proposed algorithm thành một algorithm rõ ràng hơn thay vì sự tương tác tình cờ giữa hai Top-K.

---

# 10. Critical issue #3 — query/forward/backward counting hiện không đáng tin

Đây là lỗi lớn nhất trong benchmark engine.

Trong `evaluate_attack()`:

```python
counting_model =
    CountingModel(model)
```

nhưng attack vẫn sử dụng object model cũ:

```python
output = attack.attack(x, y)
```

Attack đã được instantiate trước đó bằng:

```python
create_attack(..., model=model)
```

chứ không phải `counting_model`.

Do đó wrapper `CountingModel` hầu như **không intercept model calls của attack**.

Trong `utils.py`, counter chỉ tăng khi gọi:

```python
CountingModel.forward()
```

Nhưng object đó không thực sự nằm trong attack graph.

### Hậu quả

Các số:

```text
total_forward_evals
total_queries
```

có thể sai.

Đặc biệt nhiều external adapters đang hard-code count.

PGD0:

```python
forward_evals=self.steps
backward_evals=self.steps
```

GSE cũng tương tự.

Sigma-Zero cũng tương tự.

Nhưng một optimization step hoàn toàn có thể gọi model nhiều lần.

**Không được dùng các số này trong bảng efficiency của paper ở trạng thái hiện tại.**

### Fix đúng

Attack phải nhận:

```python
counting_model = CountingModel(model)
attack = create_attack(... model=counting_model)
```

Hoặc dùng hook ở base model.

Và cần phân biệt:

```text
forward batches
forward images
gradient evaluations
black-box queries
```

Không nên gọi tất cả là query.

---

# 11. Critical issue #4 — config "paper" hiện chỉ chạy 10 samples

Trong:

```yaml
dataset:
  samples: 10
```

Đây là smoke test, không phải paper benchmark.

Với 10 samples:

* ASR resolution = 10%
* class stratification mỗi class ~1 image
* variance cực lớn
* comparison SOTA gần như vô nghĩa.

### Nên đổi architecture config

Tách:

```text
configs/
    smoke.yaml
    debug.yaml
    paper_cifar10.yaml
    ablation.yaml
```

Trong paper:

```yaml
samples: 1000
```

tối thiểu để iteration nhanh.

Final result tốt nhất:

```yaml
samples: 10000
```

toàn bộ CIFAR-10 test set, nếu compute cho phép.

Black-box Sparse-RS 10k queries × 10k images sẽ rất đắt, nên có thể:

```text
main benchmark: 1000 common images
verification benchmark: full test for cheaper methods
```

nhưng phải ghi rõ.

---

# 12. Critical issue #5 — defense benchmark chưa có adversarial training

Original research scope của protocol nói rõ câu hỏi:

> robustness under preprocessing and adversarial-training defenses.

Nhưng implementation hiện chỉ có:

* Gaussian blur
* median
* JPEG
* TVM
* BPDA

Search repo cũng không tìm thấy pipeline adversarial training.

Do đó defense section hiện mới hoàn thành khoảng **một nửa scope**.

### Đây là gap quan trọng

Để paper hoàn chỉnh cần ít nhất:

```text
Clean model
PGD adversarially trained model
Sparse adversarially trained model
```

Tốt hơn:

```text
Standard training
PGD-Linf AT
Sparse-PGD/L0 AT
Proposed-SFA AT
```

và cross-attack matrix:

| Train defense | PGD0 | Sparse-PGD | Sparse-RS | SFA |
| ------------- | ---: | ---------: | --------: | --: |
| Standard      |      |            |           |     |
| PGD AT        |      |            |           |     |
| Sparse-PGD AT |      |            |           |     |
| SFA AT        |      |            |           |     |

Đây sẽ là experiment rất đáng giá.

---

# 13. Defense pipeline: adaptive vs oblivious là đúng hướng

`DefendedModelAdapter` có:

```python
mode = adaptive
mode = oblivious
```

và BPDA:

```python
forward:
    real defense

backward:
    identity
```

Đây là cách đánh giá defense đúng tinh thần chống gradient masking.

### Oblivious

Attack:

[
f(x)
]

evaluation:

[
f(D(x_{adv}))
]

### Adaptive

Attack trực tiếp:

[
f(D(x))
]

và BPDA nếu (D) nondifferentiable.

Tốt.

---

# 14. Nhưng defense benchmark vẫn chưa đủ adaptive

BPDA identity backward:

[
\frac{\partial D(x)}{\partial x}
\approx I
]

là baseline hợp lý nhưng không phải lúc nào cũng attack mạnh nhất.

JPEG/median/TVM cần cân nhắc:

* BPDA identity
* differentiable surrogate
* EOT nếu defense stochastic
* multiple attack restarts

Nếu chỉ dùng BPDA một lần rồi conclude robustness, reviewer có thể hỏi:

> Does the defense induce obfuscated gradients?

Nên có sanity checks.

---

# 15. Một lỗi subtle trong Gaussian blur

`gaussian_blur()` dùng:

```python
F.conv2d(... padding=radius)
```

với zero padding.

CIFAR image sẽ có artificial dark border.

Nên cân nhắc:

```python
F.pad(x, ..., mode="reflect")
```

rồi conv without zero-padding.

Median filter cũng sử dụng `F.unfold(... padding=padding)` nên border cũng được zero pad.

Điều này có thể làm clean accuracy giảm không cần thiết.

---

# 16. TVM hiện chưa thực sự là standard TV minimization

Code gọi:

```python
total_variation_minimization
```

nhưng thực chất chạy vài bước sign-gradient smoothing heuristic:

```python
x_def = x_def -
    step_size * (grad_h + grad_w)
```

Nó không solve rõ ràng objective kiểu:

[
\min_z
\frac12 |z-x|_2^2
+
\lambda TV(z)
]

Do đó nếu paper gọi là **TV Minimization defense** thì hơi mạnh quá.

Có thể gọi:

> iterative TV denoising

hoặc implement standard solver/proximal TV.

---

# 17. Checkpoint handling khá tốt

`get_model()` có một điểm rất tốt:

nếu không tìm thấy checkpoint:

```python
raise FileNotFoundError
```

thay vì chạy model random.

Đây là requirement rất quan trọng với research benchmarks.

Ngoài ra có:

```text
checkpoint SHA256
expected SHA
min clean accuracy
```

support.

Đây là một trong những phần mature nhất của repo.

---

# 18. Nhưng protocol train model chưa được implement trong repo hiện tại

Protocol mô tả:

* SGD
* lr 0.1
* momentum 0.9
* weight decay
* 200 epochs
* cosine
* checkpoint best validation

Nhưng tree hiện không có một clean training script rõ ràng.

Model được tải từ HF:

```python
HF_REPO_ID = "Cuong2004/AA"
```

Điều này chạy được nhưng reproducibility artifact thiếu:

```text
train_clean.py
```

và sau này:

```text
train_adv.py
```

Nếu reviewer muốn reproduce từ zero thì hiện chưa đủ.

---

# 19. Data pipeline nhìn chung tốt

Bạn dùng deterministic stratified split:

```python
train_test_split(
    ...,
    random_state=seed,
    stratify=all_labels
)
```

Benchmark test subset cũng stratified + sorted indices và tạo SHA256 của index sequence.

Đây là thiết kế tốt.

### Một góp ý

Evaluation transform có:

```python
Resize((32,32))
```

cho CIFAR vốn đã 32×32.

Không gây vấn đề lớn nếu PIL không thực sự đổi kích thước, nhưng về methodological cleanliness thì nên bỏ.

Chỉ:

```python
ToTensor()
```

là đủ.

---

# 20. ASR implementation đúng

Trong evaluator:

```python
success =
    clean_correct & adv_pred.ne(y)
```

và:

[
ASR =
\frac{success}{clean_correct}.
]

Đây là cách nên dùng.

Nhiều repo adversarial attack tính ASR trên toàn bộ samples khiến model clean accuracy khác nhau làm méo comparison.

Repo này tránh được lỗi đó.

---

# 21. Distortion reporting khá tốt nhưng cần statistical summaries

Hiện có:

* (L_0)
* (L_2)
* (L_\infty)
* PSNR
* SSIM
* LPIPS

nhưng chủ yếu báo:

```text
mean
median
```

### Paper nên thêm

[
mean \pm std
]

hoặc tốt hơn:

[
median,\quad Q_1,\quad Q_3
]

cho (L_0).

Và ASR cần:

```text
95% confidence interval
```

Ví dụ bootstrap hoặc Wilson interval.

---

# 22. LPIPS có dependency nhưng benchmark mặc định không sử dụng

`evaluate_attack()` nhận:

```python
lpips_fn=None
```

nhưng script benchmark không instantiate LPIPS network trước khi gọi evaluator.

Vì vậy mặc dù protocol nói LPIPS, thực tế `paper.yaml` run hiện **không produce LPIPS**.

Đây là một protocol ↔ implementation mismatch.

---

# 23. Proposed feature loss cần normalization tốt hơn

Hiện:

```python
feat_l =
    sum((feat_adv - feat_clean)^2) / feature_dim
```

Nó normalization theo số feature dimensions, nhưng magnitude vẫn phụ thuộc mạnh vào:

* layer
* model architecture
* activation scale.

WRN và ResNet có thể có loss scale rất khác nhau.

### Nên cân nhắc

Relative feature disruption:

[
L_f =
\frac{
|\phi(x_{adv})-\phi(x)|_2^2
}{
|\phi(x)|_2^2+\epsilon
}
]

hoặc cosine:

[
1-\cos(
\phi(x_{adv}),
\phi(x)
)
]

Điều này tốt hơn cho cross-backbone generalization.

---

# 24. CE loss có thể không phải objective attack tối ưu nhất

SFA đang maximize:

[
CE(f(x),y)
]

via gradient ascent.

Sparse attack thường có lợi nếu dùng margin loss:

[
L_{margin}
==========

\max_{j\neq y} z_j-z_y
]

hoặc DLR-like objective.

CE có thể saturate và support ranking dựa vào CE gradient chưa chắc tốt.

Đây là một ablation quan trọng:

```text
CE
CW margin
DLR
```

Nếu proposed method có kết quả tốt hơn đơn thuần nhờ loss mạnh hơn baseline thì cũng phải tránh unfair comparison.

---

# 25. Proposed pruning là ý tưởng tốt nhưng chi phí rất lớn

Sau success, code thử xóa từng active pixel:

```python
for coord in active_coords:
    test prediction
```

lặp tối đa 5 passes.

Nếu (K=64), số forward extra có thể gần:

[
64+63+62+\cdots
]

tùy mức pruning.

Điều này khiến:

> ASR/L0 tốt hơn

nhưng runtime/query cost cao hơn đáng kể.

Do query accounting hiện chưa chính xác, nguy cơ benchmark sẽ vô tình hide chi phí pruning.

### Cần report riêng

```text
Optimization queries
Pruning queries
Total queries
```

và ablation:

```text
SFA w/o pruning
SFA + pruning
```

---

# 26. Greedy pruning order chưa chắc optimal

Code sort pixel bằng perturbation magnitude tăng dần:

```python
sort by smallest magnitude first
```

Nhưng magnitude thấp không đồng nghĩa contribution thấp.

Có thể dùng:

[
Importance_i =
|\delta_i \odot \nabla_i L|
]

hoặc leave-one-out score.

Một improved variant khá tự nhiên:

```text
support pruning by marginal adversarial contribution
```

Đây có thể là một phần proposed-method mới đáng nghiên cứu.

---

# 27. Baseline provenance: rất tốt về ý thức reproducibility

`THIRD_PARTY.md` ghi:

* upstream repo
* pinned commit
* exact tree / core file verified
* license state
* adapter

Đây là mức provenance tốt hơn rất nhiều research repo thông thường.

Ví dụ Sparse-RS được ghi exact tree với pinned commit.

Đây là thứ nên giữ cho artifact submission.

---

# 28. Nhưng THIRD_PARTY.md hiện có stale paths

Tài liệu vẫn nói adapter ở các path kiểu:

```text
src/attacks/adapters/...
```

trong khi architecture hiện là:

```text
src/aa/attacks/external/...
```

Tài liệu còn tham chiếu:

```text
docs/baseline_validation.md
```

nhưng file này không xuất hiện trong tree mình đọc.

Đây là dấu hiệu refactor đã xong code nhưng documentation cleanup chưa hoàn tất.

Không nghiêm trọng về algorithm, nhưng artifact submission sẽ bị reviewer để ý.

---

# 29. External adapters cần validation mạnh hơn

Ví dụ PGD0 convert:

```python
PyTorch BCHW
→ CPU NumPy BHWC
→ official attack
→ PyTorch BCHW
```

Đây có thể đúng với upstream API nhưng cần integration test xác nhận:

[
L_0 \le K
]

và input range:

[
x_{adv}\in[0,1].
]

Hiện `test_attacks.py` chỉ chạy:

```python
fgsm
pgd
ours
```

không chạy external attacks.

Đây là thiếu sót đáng kể.

---

# 30. Test coverage chưa đủ

Tests hiện verify chủ yếu:

* output shape

* clipping

* proposed (L_0\le K)

* basic benchmark counts

### Cần thêm integration tests

Cho từng attack:

```text
test_output_range
test_l0_constraint
test_determinism
test_clean_correct_condition
test_batch_size_1
test_batch_size_N
test_CPU
test_CUDA
```

Đặc biệt:

```text
CornerSearch
PGD0
Sparse-PGD
Sparse-RS
SparseFool
Sigma-Zero
GSE
```

Cũng nên có:

```python
assert compute_spatial_l0(x_adv-x) <= K
```

cho mọi **budgeted external attack**.

---

# 31. Registry filtering kwargs là tiện nhưng có nguy cơ silent configuration errors

`create_attack()` làm:

```python
filtered_kwargs = {
    k:v for k,v in kwargs.items()
    if k in valid_params
}
```

Điều này tiện nhưng nguy hiểm.

Nếu config typo:

```yaml
n_query: 10000
```

thay vì:

```yaml
n_queries: 10000
```

nó silently bỏ parameter.

Benchmark vẫn chạy nhưng bằng default.

### Nên đổi

Trong paper mode:

```python
unknown = kwargs.keys() - valid_params
if unknown:
    raise ValueError(...)
```

Research benchmarks nên **fail loudly**.

---

# 32. Exception handling benchmark cũng đang quá permissive

Script làm:

```python
except Exception as e:
    save {"error": ...}
    continue
```

Smoke mode thì tốt.

Paper mode thì không.

Ví dụ 3 baseline fail nhưng script vẫn tạo JSON hoàn chỉnh. Sau đó plot script có thể vô tình bỏ missing methods.

### Nên có:

```yaml
strict: true
```

Và nếu paper benchmark:

```python
raise
```

ngay khi một required baseline fail.

---

# 33. Dense attacks không trực tiếp comparable với sparse attacks

FGSM/BIM/PGD là (L_\infty)-bounded dense attacks.

Sparse attacks là (L_0)-bounded.

Protocol gọi dense methods là references, điều này đúng.

Trong paper không nên đưa chúng cùng bảng và xếp hạng theo ASR như thể cùng threat model.

Nên tách:

### Table A — dense sanity references

[
L_\infty = 8/255
]

### Table B — sparse main benchmark

[
L_0 \le K
]

---

# 34. Paper benchmark hiện chưa có repeated seeds

Config chỉ:

```yaml
seed: 42
```

Đối với:

* random-start PGD
* Sparse-RS
* stochastic algorithms

single seed không đủ mạnh.

Nên chạy:

[
s\in{0,1,2,3,4}
]

ít nhất 3 seeds cho stochastic attacks.

Reporting:

[
mean\ ASR \pm std
]

---

# 35. Proposed attack thiếu random restart

Current SFA starts:

```python
delta = zeros
```

Không có restart.

Sparse optimization cực non-convex.

Một improved version nên có:

```text
n_restarts = 5
```

với:

* random support
* gradient support
* mixed support

rồi success-first/min-(L_0) selection.

Đây nhiều khả năng sẽ tăng attack strength rõ.

---

# 36. Proposed method có một conceptual tension

Document formulation là:

[
\min |\delta|_0
]

nhưng algorithm thực tế nhận cố định:

```python
k
```

và solve gần hơn:

[
\max L(x+\delta)
\quad
s.t.
\quad
|\delta|_0\le K.
]

sau đó mới greedy-prune.

Hai formulation này không giống hoàn toàn.

### Paper nên mô tả đúng

Main attack:

[
\max_\delta \mathcal L
\quad
s.t.
\quad
|\delta|_{0,\mathrm{spatial}}\le K
]

và pruning là stage 2:

[
\min_{\tilde\delta}
|\tilde\delta|_0
\quad
s.t.
\quad
supp(\tilde\delta)\subseteq supp(\delta^*)
]

và vẫn misclassify.

Đây sẽ chính xác hơn.

---

# 37. SSIM implementation nên validate against standard library

Repo implement custom SSIM bằng Gaussian window.

Không hẳn sai, nhưng paper metric nên validate bằng:

* `torchmetrics`
* `skimage.metrics.structural_similarity`

trên một test suite.

Custom metrics là một nguồn reviewer skepticism không cần thiết.

---

# 38. Reproducibility metadata tốt

Result JSON lưu:

```text
config
device
sample_indices_hash
checkpoint SHA256
git commit
dirty status
Python
PyTorch
platform
device name
```

Đây là rất tốt.

Mình sẽ thêm:

```text
CUDA version
cuDNN version
GPU count
hostname optional
third-party commit hashes
dataset fingerprint
attack source version
```

---

# 39. Roadmap hơi quá lạc quan

Roadmap đánh dấu tất cả Phase 1–9 là completed, kể cả benchmark engine và test suite.

Theo code hiện tại mình sẽ không gọi Phase 7–8 hoàn thành theo nghĩa paper-ready.

Nên đổi trạng thái:

```text
Phase 7 — infrastructure complete, validation pending
Phase 8 — core tests complete, external integration pending
```

Điều này phản ánh repo chính xác hơn.

---

# 40. Những vấn đề Critical cần sửa trước khi chạy experiment chính

Mình xếp như sau:

### P0 — phải sửa trước paper benchmark

1. **Fix forward/query/backward accounting.**
2. **Tăng paper sample size khỏi 10.**
3. **Thêm adversarial training pipeline.**
4. **Validate tất cả external baseline adapters.**
5. **Strict config validation; không silently drop kwargs.**
6. **Strict benchmark failure mode.**
7. **Implement LPIPS thật nếu protocol claim LPIPS.**
8. **Thêm stochastic repeated runs / seeds.**
9. **Document đúng cost của support pruning.**
10. **Tách smoke config khỏi paper config.**

---

# 41. P1 — cải thiện proposed method

Mình sẽ ưu tiên nghiên cứu:

### A. Dynamic support replacement

Thay vì static Top-K mỗi iteration:

[
S_{t+1}
=======

Keep(S_t,q)
\cup
Explore(Score_t,K-q).
]

### B. Margin/DLR attack loss

Thay CE bằng:

[
L =
z_{max\neq y}-z_y
+
\lambda_f L_f.
]

### C. Relative feature disruption

[
L_f =
\frac{
|\phi(x_{adv})-\phi(x)|^2
}{
|\phi(x)|^2+\epsilon
}.
]

### D. Multiple restarts

```text
gradient support
random support
hybrid support
```

### E. Contribution-aware pruning

Thay magnitude-only pruning.

### F. Progressive support minimization

```text
64 → 32 → 16 → 8 → ...
```

với warm start.

---

# 42. Ablation mình đề xuất

Hiện docs mới định nghĩa:

```text
A0 base
A1 feature
A2 interaction
A3 full + pruning
```

Mình nghĩ chưa đủ.

Nên dùng:

| Variant | Feature | Interaction | Dynamic support | Pruning |
| ------- | ------: | ----------: | --------------: | ------: |
| A0      |       ✗ |           ✗ |               ✗ |       ✗ |
| A1      |       ✓ |           ✗ |               ✗ |       ✗ |
| A2      |       ✗ |           ✓ |               ✗ |       ✗ |
| A3      |       ✓ |           ✓ |               ✗ |       ✗ |
| A4      |       ✓ |           ✓ |               ✓ |       ✗ |
| Full    |       ✓ |           ✓ |               ✓ |       ✓ |

Và riêng interaction:

```text
none
smoothing
CPA
FCSA
HSA
```

Nếu CPA thắng consistently thì hãy chọn CPA làm final proposed.

---

# 43. Main experiment matrix mình khuyến nghị

## Dataset

Primary:

```text
CIFAR-10
```

Generalization:

```text
CIFAR-100
```

sau.

## Models

```text
ResNet-18
WRN-28-10
```

Protocol đã định hướng đúng như vậy.

## Sparse budget

[
K=
1,2,4,8,16,32,64.
]

## Metrics

Main:

```text
ASR@K ↑
Robust Accuracy ↓
median L0 ↓
Queries ↓
Runtime ↓
```

Secondary:

```text
L2
Linf
PSNR
SSIM
LPIPS
```

---

# 44. Bảng chính nên trông như thế này

### Table 1 — Sparse Attack Effectiveness

| Method       | K=1 |  2 |  4 |  8 | 16 | 32 | 64 |
| ------------ | --: | -: | -: | -: | -: | -: | -: |
| CornerSearch |     |    |    |    |    |    |    |
| PGD0         |     |    |    |    |    |    |    |
| Sparse-PGD   |     |    |    |    |    |    |    |
| Sparse-RS    |     |    |    |    |    |    |    |
| SparseFool   |     |    |    |    |    |    |    |
| Sigma-Zero   |     |    |    |    |    |    |    |
| GSE          |     |    |    |    |    |    |    |
| **SFA**      |     |    |    |    |    |    |    |

Cell = ASR%.

---

# 45. Bảng efficiency riêng

| Attack | ASR@16 | Median L0 | Queries/img | Grad evals | Time/img |
| ------ | -----: | --------: | ----------: | ---------: | -------: |

Không nên trộn efficiency với main ASR table.

---

# 46. Defense table

Adaptive attacks mới là con số quan trọng:

| Defense       | Clean Acc | PGD0 | Sparse-RS | SFA |
| ------------- | --------: | ---: | --------: | --: |
| None          |           |      |           |     |
| Blur          |           |      |           |     |
| Median        |           |      |           |     |
| JPEG          |           |      |           |     |
| TV denoise    |           |      |           |     |
| PGD AT        |           |      |           |     |
| Sparse-PGD AT |           |      |           |     |
| SFA AT        |           |      |           |     |

Cell = defended robust accuracy hoặc ASR, nhưng phải thống nhất.

---

# 47. Một experiment rất đáng làm: cross-attack adversarial training

Nếu paper muốn attack + defense cùng một story, đây có thể là experiment mạnh nhất.

Train bằng:

```text
PGD-Linf
Sparse-PGD
SFA
```

rồi test bằng:

```text
PGD0
Sparse-RS
Sigma-Zero
SFA
```

Bạn sẽ có transfer matrix:

[
R_{ij}
======

RobustAccuracy(
TrainAttack_i,
EvalAttack_j
).
]

Từ đó trả lời được:

> SFA chỉ mạnh như một attack hay perturbations của nó còn tạo adversarial training tốt hơn?

Đây là research question đáng kể.

---

# 48. Về code quality

Code nhìn chung:

* readable,
* typing có sử dụng,
* module separation tốt,
* ít global state,
* explicit errors ở checkpoint/feature extractor.

Nhưng chưa có:

```text
ruff
black
mypy
pre-commit
CI
```

trong phần tree mình xem.

Cho artifact quality, nên thêm GitHub Actions:

```text
pytest
ruff
```

mỗi commit.

---

# 49. Về dependency reproducibility

`pyproject.toml` dùng:

```text
torch>=2.0
torchvision>=0.15
datasets>=2.10
...
```

Đây là quá rộng cho reproducibility.

Research artifact nên có thêm:

```text
requirements-lock.txt
```

hoặc:

```text
uv.lock
poetry.lock
conda environment.yml
```

để replicate environment exact.

---

# 50. Đánh giá final proposed method hiện tại

Nếu chỉ nhìn concept:

> feature-space disruption + local interaction-aware spatial ranking + exact (L_0) projection + support pruning

thì **có tiềm năng**.

Nhưng novelty hiện chưa đủ được chứng minh chỉ bằng code.

Điều paper cần chứng minh là:

[
ASR_{SFA}(K)

>

ASR_{SparseRS}(K),
ASR_{PGD0}(K),
ASR_{SPGD}(K)
]

trong phần đáng kể của budget range, đặc biệt:

[
K\le16.
]

Và ideally:

[
Queries_{SFA}
<
Queries_{strong\ baselines}
]

hoặc:

[
L0_{SFA}
<
L0_{baselines}
]

ở matched ASR.

Nếu chỉ thắng sau pruning với hàng trăm extra forwards thì claim cần framing khác.

---

# 51. Điểm mình đánh giá rất cao

Có 5 quyết định thiết kế mình nghĩ nên giữ nguyên:

1. **Spatial (L_0) thay vì channel-wise (L_0).**
2. **Conditional ASR trên clean-correct samples.**
3. **Same deterministic sample indices cho mọi attack.**
4. **Official baseline adapters + provenance tracking.**
5. **Adaptive BPDA evaluation cho preprocessing defense.**

Đó là nền tảng research benchmark tốt.

---

# 52. Điểm nguy hiểm nhất nếu chạy paper ngay bây giờ

Nếu hiện tại chạy:

```bash
python scripts/attack_benchmark.py \
    --config configs/paper.yaml
```

thì có thể thu được một JSON rất đẹp.

Nhưng mình sẽ **không dùng trực tiếp con số đó cho paper**, vì:

* chỉ có 10 samples;
* evaluation accounting chưa reliable;
* external adapters chưa có sufficient integration tests;
* stochastic variance chưa được đo;
* LPIPS protocol chưa được thực thi;
* adversarial training defense còn thiếu;
* pruning làm tăng query cost nhưng accounting chưa đủ chính xác.

---

# 53. Mức độ hoàn thiện thực tế

Mình sẽ gọi repository hiện tại là:

> **Strong research prototype / benchmark infrastructure v0.7**

chứ chưa phải:

> **final reproducible paper artifact v1.0**.

Roadmap đang đánh dấu Phase 1–9 complete.

Về software structure thì hợp lý.

Về scientific validation thì vẫn còn một phase rất quan trọng:

```text
Phase 10
Scientific Validation & Paper Benchmark
```

mình đề xuất thêm.

---

# 54. Phase tiếp theo mình khuyến nghị

Thứ tự làm sẽ là:

```text
P0.1 Fix exact evaluation/query accounting
      ↓
P0.2 Validate every official baseline
      ↓
P0.3 Split smoke/paper configs
      ↓
P0.4 Run ≥1000 common samples
      ↓
P0.5 Add repeated seeds
      ↓
P0.6 Improve SFA support optimization
      ↓
P0.7 Run full ablations
      ↓
P0.8 Implement adversarial training
      ↓
P0.9 Adaptive defense evaluation
      ↓
P0.10 Final tables + statistics
```

Sau đó mới freeze proposed method.

---

## Kết luận

**Nền tảng của AA_2 khá tốt.** Project đã giải quyết đúng nhiều vấn đề mà research adversarial-attack repos thường làm thiếu: threat model rõ, common samples, conditional ASR, registry chung, official implementation provenance, checkpoint hashing và adaptive preprocessing defenses.

Nhưng vấn đề hiện tại không còn nằm chủ yếu ở “code organization”; nó nằm ở **scientific rigor**. Ba thứ mình ưu tiên tuyệt đối là **sửa measurement/accounting**, **validate baseline fairness**, và **nâng SFA từ heuristic support-ranking thành một sparse optimization algorithm có formulation rõ hơn**. Sau đó mới đáng chạy benchmark lớn và bắt đầu viết main claim cho paper.

Nếu xét riêng mục tiêu Sparse AA + defense của dự án, repo này đã có một nền móng tốt để phát triển, nhưng **chưa đủ bằng chứng để kết luận proposed method là SOTA hoặc defense là robust** ở trạng thái hiện tại.
