Tôi đã đọc cấu trúc repo và các phần cốt lõi gồm protocol, config benchmark, data/model loading, attack registry, dense baselines, proposed attack, defense pipeline, metrics/evaluation và tests. Đánh giá tổng thể: **ý tưởng kiến trúc nghiên cứu tốt và khá đúng hướng cho một paper Sparse Adversarial Attack, nhưng code hiện tại chưa đủ tin cậy để dùng số liệu cho paper** vì đang có một số lỗi correctness nghiêm trọng và vài vấn đề fairness trong benchmark.

## 1. Tổng quan kiến trúc

Repo được chia khá sạch:

```text
AA_2/
├── configs/
│   └── paper.yaml
├── docs/
│   ├── protocol.md
│   ├── proposed_method.md
│   ├── roadmap.md
│   └── upgrade.md
├── scripts/
│   ├── attack_benchmark.py
│   └── defense_benchmark.py
├── src/aa/
│   ├── attacks/
│   │   ├── base.py
│   │   ├── dense.py
│   │   ├── proposed.py
│   │   ├── registry.py
│   │   └── external/
│   ├── benchmark.py
│   ├── data.py
│   ├── defenses.py
│   ├── metrics.py
│   ├── models.py
│   └── utils.py
├── tests/
└── third_party/
```

README định vị repo là một benchmark reproducible cho **pixel-sparse adversarial attacks**, với protocol làm "single source of truth". Đây là cách tổ chức phù hợp cho research code: method được tách khỏi evaluation, external attacks có adapters riêng, protocol/documentation nằm ngoài code.

Tôi chấm phần tổ chức repository khoảng **8/10**.

---

# 2. Experimental protocol: phần tốt nhất của repo

`docs/protocol.md` được viết khá tốt.

Threat model được định nghĩa đúng theo **spatial L0**:

[
|\delta|_{0,\text{spatial}}
===========================

\sum_{h,w}
\mathbf{1}\left[
\max_c |\delta_{c,h,w}|>\tau
\right]
]

với (\tau=10^{-5}).

Budget:

[
K\in{1,2,4,8,16,32,64}.
]

Đây là một lựa chọn hợp lý cho CIFAR-10.

Protocol còn phân chia attacks khá đúng về mặt methodology:

**Dense references**

* FGSM
* BIM
* PGD

**Budgeted sparse**

* CornerSearch
* PGD0
* Sparse-PGD
* Sparse-RS

**Minimal-support**

* SparseFool
* Sigma-Zero
* GSE

**Proposed**

* SparseFeatureAttack

Đây là điểm rất tốt vì **minimal-support attack không nên được benchmark giống hệt fixed-budget attack**.

Repo đã nhận ra điều này và `attack_benchmark.py` chạy minimal attacks một lần rồi derive `ASR@K` từ L0 thực tế.

Đây là thiết kế đúng hơn rất nhiều so với việc ép SparseFool/Sigma-Zero chạy lại cho từng K.

---

# 3. Dataset sampling và reproducibility

Phần này tương đối tốt.

CIFAR train được chia:

* 40k training
* 10k validation

bằng stratified split cố định.

Benchmark subset của test set cũng dùng stratified sampling với seed, sau đó hash danh sách sample indices.

Kết quả benchmark còn ghi:

* config
* device
* sample hash
* checkpoint SHA256
* git reproducibility info

### Đây là điểm rất đáng giữ.

Đối với paper adversarial ML, việc tất cả attacks chạy trên **chính xác cùng tập sample** quan trọng hơn nhiều người nghĩ.

### Nhưng `samples: 100` hiện tại là quá thấp

`paper.yaml` đang để:

```yaml
dataset:
  samples: 100
```

100 samples chỉ phù hợp:

* development;
* debugging;
* sanity benchmark.

**Không đủ cho main table của paper.**

Ví dụ ASR = 70% trên 100 ảnh có variance rất lớn. Chỉ 3–5 ảnh thay đổi đã làm conclusion thay đổi đáng kể.

Tôi đề nghị:

```yaml
development:
    samples: 100

validation_experiments:
    samples: 1000

paper_main:
    samples: 10000
```

Nếu một số black-box attacks như Sparse-RS quá đắt thì có thể dùng 1000–2000 samples, nhưng phải báo rõ protocol.

---

# 4. Lỗi P0: Benchmark hiện tại có khả năng crash

Đây là vấn đề nghiêm trọng nhất tôi thấy.

Trong `metrics.py`:

```python
class BatchMetrics(NamedTuple):
    clean_correct: torch.Tensor
    success: torch.Tensor
    l0: torch.Tensor
    l2: torch.Tensor
    linf: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor
```

Nhưng phía dưới lại:

```python
return BatchMetrics(
    clean_correct=clean_correct,
    adv_correct=adv_correct,
    success=success,
    ...
    lpips=lpips
)
```

Tức là constructor nhận thêm:

```text
adv_correct
lpips
```

nhưng hai field này **không tồn tại trong NamedTuple**.

Kết quả sẽ là dạng:

```text
TypeError:
BatchMetrics.__new__() got an unexpected keyword argument 'adv_correct'
```

hoặc tương đương.

Trong khi `benchmark.py` lại tiếp tục truy cập:

```python
batch_m.adv_correct
batch_m.lpips
```

### Đây là P0 blocker.

Nên sửa:

```python
class BatchMetrics(NamedTuple):
    clean_correct: torch.Tensor
    adv_correct: torch.Tensor
    success: torch.Tensor
    l0: torch.Tensor
    l2: torch.Tensor
    linf: torch.Tensor
    psnr: torch.Tensor
    ssim: torch.Tensor
    lpips: Optional[torch.Tensor]
```

Điều đáng chú ý hơn: `tests/test_benchmark.py` thực sự gọi `evaluate_attack()`.

Nếu test suite được chạy với code hiện tại, lỗi này đáng lẽ phải xuất hiện.

Điều này làm tôi nghi ngờ **CI/test chưa được chạy liên tục trên commit hiện tại**.

---

# 5. Lỗi metric: Conditional Robust Accuracy đang tính sai denominator

Trong `benchmark.py`:

```python
cond_robust_acc = (
    100.0 * (clean_correct_count - succ_count) / total_samples
)
```

Nếu gọi metric là **conditional robust accuracy**, denominator phải là số clean-correct samples:

[
CRA
===

\frac{N_{\text{clean correct}}-N_{\text{success}}}
{N_{\text{clean correct}}}.
]

Không phải:

[
\frac{N_{\text{clean correct}}-N_{\text{success}}}
{N_{\text{all}}}.
]

Ví dụ:

```text
1000 images
clean correct = 900
attack succeeds = 450
```

ASR:

[
450/900=50%.
]

Conditional robust accuracy phải:

[
450/900=50%.
]

Code hiện tại trả:

[
450/1000=45%.
]

### Vì thế hai metric đang không complementary:

Expected:

[
ASR + CRA = 100%.
]

Code hiện tại thì không.

Đây là **P0/P1 statistical correctness issue**.

Đáng sửa thành:

```python
conditional_robust_accuracy = (
    100.0
    * (clean_correct_count - succ_count)
    / clean_correct_count
)
```

Trong khi full-set robust accuracy:

```python
adv_corr_cat.float().mean()
```

là hợp lý.

---

# 6. Proposed Method: concept khá thú vị

Method hiện tại là:

> Feature-Guided Collaborative Sparse Adversarial Attack with Support Pruning.

Pipeline thực tế gần như:

[
x
\rightarrow
\nabla_x\mathcal L
\rightarrow
\text{interaction score}
\rightarrow
\text{TopK support}
\rightarrow
\text{gradient update}
\rightarrow
P_{L_0(K)}
\rightarrow
\text{success check}
\rightarrow
\text{support pruning}.
]

Có 3 thành phần chính.

### Feature-guidance

Loss:

[
L =
L_{CE}
+
\lambda L_{feature}.
]

Feature loss hiện được normalize theo feature dimension:

```python
feat_l =
    feat_diff.pow(2)
    .flatten(1)
    .sum(dim=1)
    / feat_dim
```

Đây là quyết định hợp lý vì tránh magnitude scale tăng theo kích thước representation.

---

## 7. CPA interaction là hướng có tiềm năng

Mode mặc định:

```yaml
interaction_mode: cpa
```

CPA đang định nghĩa pixel importance gần dạng:

[
I_i =
|g_i|*1+
\lambda
\sum*{j\in N(i)}
\operatorname{ReLU}
(\cos(g_i,g_j))
|g_j|_1.
]

Ý tưởng trực giác tốt:

> Không chỉ chọn pixel có gradient lớn mà chọn pixel nằm trong vùng gradient phối hợp cùng nhau.

Nó tạo differentiation khá rõ so với vanilla top-k gradient attack.

### Đây có thể là contribution thực sự.

Tuy nhiên hiện implementation có một lỗi subtle.

Neighborhood dùng:

```python
torch.roll(...)
```

`torch.roll` tạo **circular boundary**.

Ví dụ pixel:

```text
(0,0)
```

sẽ coi pixel:

```text
(31,31)
```

là hàng xóm.

Về spatial image topology, điều đó không đúng.

Nó tạo artificial cooperation giữa các cạnh đối diện của ảnh.

Nên dùng padding + shifted slicing thay vì `torch.roll`.

Đây là **P1 methodological bug**, đặc biệt nếu CPA là contribution chính.

---

# 8. FCSA implementation hiện tại bị degenerative

Đây là lỗi lớn thứ hai trong proposed method.

Trong `fcsa`:

```python
indiv_contrib = grad_mag * grad_max

patch_contrib =
    F.avg_pool2d(indiv_contrib, ...) * 9

sum_indiv =
    F.avg_pool2d(indiv_contrib, ...) * 9

synergy = F.relu(patch_contrib - sum_indiv)
```

Nhưng:

```python
patch_contrib == sum_indiv
```

một cách **chính xác**.

Do đó:

[
synergy = ReLU(0)=0.
]

Và FCSA luôn rút gọn về:

[
score = indiv_contrib.
]

Tức là component "Functional Coalition Synergy" hiện tại **không thực hiện coalition synergy nào cả**.

Nếu ablation paper chạy:

```text
CPA
FCSA
HSA
Smoothing
```

thì FCSA result hiện tại không đại diện cho method mà tên gọi mô tả.

### P0 nếu FCSA xuất hiện trong paper.

### P1 nếu chỉ là experimental branch.

---

# 9. Support pruning: rất đáng giữ

Sau khi tìm được adversarial example, proposed method thử bỏ từng modified pixel theo magnitude nhỏ nhất:

```python
pruned_delta[:, :, h, w] = 0
```

và giữ việc xóa nếu prediction vẫn sai.

Đây là phần rất có giá trị với sparse attacks.

Nó biến method từ:

> "find successful point under K"

thành:

> "find success under K, sau đó minimize actual support".

Vì vậy có thể report đồng thời:

* ASR@K
* achieved L0
* median successful L0.

Đó là thiết kế paper-friendly.

### Nhưng cần fairness

Nếu `ours` có post-hoc pruning còn các budgeted baselines không được equivalent refinement, thì không nên so sánh:

```text
median achieved L0
```

một cách trực tiếp rồi tuyên bố method efficient hơn.

Có hai cách fair:

**A.** chỉ dùng pruning như một phần algorithm được tuyên bố rõ;

hoặc

**B.** cung cấp generic pruning refinement cho tất cả attacks trong một secondary comparison.

Tôi nghiêng về A cho main method, nhưng phải nói rõ cost của pruning.

---

# 10. Proposed attack dừng ngay khi success

Code có:

```python
if fooled_mask.all():
    break
```

và những sample đã success được freeze:

```python
fooled_mask
```

Điều này tốt về computational efficiency.

Nhưng nó có một hệ quả:

`best_adv` không tiếp tục optimize margin sau success.

Do đó method được optimize theo triết lý:

> **success-first / minimal computation**

chứ không phải:

> maximize adversarial confidence within K.

Với L0 attack, đây không phải vấn đề; thậm chí có thể là ưu điểm.

Nhưng paper nên mô tả rõ objective.

---

# 11. Một vấn đề nghiêm trọng ở defense + proposed method

`SparseFeatureAttack` yêu cầu tìm feature layer:

```python
layer4
block3
features
```

nếu `feature_guidance=True`.

Nhưng defense benchmark wrap network thành:

```python
DefendedModelAdapter(base_model, ...)
```

Wrapper này có:

```python
self.model = ...
```

nhưng bản thân `DefendedModelAdapter` không có:

```text
layer4
block3
features
```

Vì FeatureExtractorAdapter được gọi với:

```python
required=self.feature_guidance
```

default = `True`, proposed attack trên defended model rất có thể raise:

```text
ValueError:
Feature guidance requested ... no valid intermediate feature layer found
```

### Hệ quả

Trong defense benchmark:

```yaml
defense_attacks:
  - pgd0
  - sparse_rs
  - ours
```

thì **`ours` có khả năng fail trên toàn bộ defense experiments**.

Đây là P0 cho defense evaluation.

Giải pháp tốt hơn:

Feature adapter cần unwrap:

```python
base_model = getattr(model, "model", model)
```

hoặc hỗ trợ nested module path như:

```text
model.layer4
```

thay vì chỉ `hasattr(self.model, "layer4")`.

---

# 12. Defense design: adaptive vs oblivious là đúng hướng

`DefendedModelAdapter` có hai mode:

### Adaptive

Attack nhìn xuyên qua preprocessing.

Differentiable defense:

```python
defended_x = defense.defend(x)
```

Non-differentiable defense:

```python
BPDAFunction.apply(x, defense)
```

### Oblivious

Attack chạy vào undefended base model nhưng evaluation chạy defended pipeline.

Đây là một distinction rất quan trọng.

Nhiều paper defense yếu chỉ report:

```text
attack → preprocessing → model
```

mà không adaptive attack, dẫn đến gradient masking.

Repo này đã chủ động triển khai BPDA, đây là **một điểm mạnh đáng kể**.

---

# 13. Tuy nhiên defense suite hiện chưa đủ cho research claim

Có:

* Gaussian blur
* median filter
* JPEG
* TV minimization

Đây đều là **preprocessing defenses**.

Trong research scope, protocol nói muốn đánh giá:

> preprocessing and adversarial-training defenses.

Nhưng trong source hiện tại tôi không thấy implementation adversarial training tương ứng; code search cho training/adversarial-training không trả về module triển khai.

Do đó defense pipeline hiện mới hoàn thành khoảng **một nửa claim**.

Tôi sẽ chia defense paper thành:

```text
D0 Clean model
D1 Gaussian
D2 Median
D3 JPEG
D4 TVM
D5 Dense adversarially trained
D6 Sparse adversarially trained
D7 Hybrid/adaptive sparse AT
```

Đặc biệt nếu paper nói về sparse attacks thì **sparse adversarial training** có ý nghĩa hơn chỉ PGD-L∞ adversarial training.

---

# 14. Defense preprocessing có border artifacts

Gaussian:

```python
F.conv2d(..., padding=radius)
```

median:

```python
F.unfold(..., padding=padding)
```

Default padding là zero padding.

Ở CIFAR 32×32, border pixels chiếm phần không nhỏ.

Preprocessing defense vì thế đưa artificial dark boundary context.

Tốt hơn nên cân nhắc:

```text
reflect padding
replicate padding
```

và ghi rõ trong protocol.

---

# 15. TVM hiện không thực sự là standard TV minimization

Function tên:

```python
total_variation_minimization
```

nhưng implementation thực chất là vài bước:

```python
x_def =
x_def - step_size * (grad_h + grad_w)
```

với sign differences.

Đây gần hơn với một heuristic TV smoothing iteration hơn là giải đúng optimization problem TV denoising kiểu:

[
\min_z
\frac12|z-x|_2^2+\lambda TV(z).
]

Nếu gọi trong paper là **TVM**, reviewer có thể hỏi:

> Đây có phải implementation tương ứng literature baseline không?

Tôi sẽ đổi naming thành:

```text
TV smoothing
```

hoặc implement standard TV denoising / cite exact algorithm.

---

# 16. Dense attacks khá chuẩn nhưng chưa phải robust reference implementation

FGSM/BIM/PGD implementation cơ bản đúng.

PGD:

```python
x_adv += alpha * sign(grad)
eta = clamp(x_adv - x, -eps, eps)
```

và clamp pixel range.

Ổn.

Một issue nhỏ:

```python
loss.backward()
```

làm gradient parameters của model accumulate qua iterations vì không có:

```python
model.zero_grad()
```

Điều này thường **không làm sai input gradient** ở case này, nhưng:

* tốn memory;
* dirty parameter grads;
* làm code khó reason hơn.

Nên dùng:

```python
grad = torch.autograd.grad(
    loss, x_adv, only_inputs=True
)[0]
```

thay vì `.backward()`.

Proposed attack cũng nên dùng pattern này.

---

# 17. Attack cost accounting hiện chưa đủ đáng tin

Protocol muốn report:

* wall-clock;
* forward evaluations;
* backward evaluations;
* queries.

Đây là đúng.

Nhưng PGD0 adapter trả:

```python
forward_evals=self.steps
backward_evals=self.steps
```

Đây là **assumption từ hyperparameter**, không phải measurement thực tế.

Official attack có thể:

* gọi model nhiều hơn 1 lần/step;
* thực hiện restart;
* evaluate success;
* làm auxiliary forwards.

Trong khi proposed attack tự đếm actual model calls.

Do vậy comparison:

```text
ours = X forward calls
PGD0 = steps
```

có thể không apples-to-apples.

### Khuyến nghị

Wrap model bằng một counting proxy ở **bên trong tất cả attacks**, rồi count actual:

```text
number of samples sent through model
```

không dựa trên attack-reported estimates.

---

# 18. Counting model trong benchmark dường như chưa được tận dụng đúng

`evaluate_attack()` tạo:

```python
counting_model =
    model if isinstance(model, CountingModel)
    else CountingModel(model)
```

nhưng sau đó attack đã được tạo trước đó với **original `model`**, không phải `counting_model`.

Và evaluation cũng gọi:

```python
evaluate_batch(model, ...)
```

không phải necessarily counting wrapper.

Do đó fallback:

```python
counting_model.forward_calls
counting_model.samples_evaluated
```

không đáng tin nếu wrapper chưa thực sự nằm trên attack inference path.

Đây là một architectural mismatch.

Tôi sẽ refactor API thành:

```python
counting_model = CountingModel(model)
attack = create_attack(name, model=counting_model)
evaluate_attack(counting_model, attack, ...)
```

và metrics forwards cần được tách khỏi attack-cost counters.

---

# 19. Query metric đang trộn semantics

Code:

```python
total_queries += getattr(
    output,
    "queries",
    counting_model.samples_evaluated
)
```

Nhưng:

* gradient attacks dùng forward/backward;
* black-box attacks dùng queries;
* model batch size ảnh hưởng calls;
* `samples_evaluated` khác `forward_calls`.

Paper cần định nghĩa chính xác:

[
Q_i = \text{number of model evaluations for sample }i
]

hay:

[
Q = \text{number of batch forward API calls}.
]

Hai thứ này rất khác nhau.

Tôi khuyến nghị report:

```text
Forward examples
Backward examples
Black-box queries/image
Wall-clock/image
```

thay vì chỉ `forward_evals`.

---

# 20. Minimal-support ASR curve là design rất tốt

`derive_minimal_asr_curve()` dùng:

```python
k_succ = success & (l0 <= k)
```

tương ứng:

[
ASR@K =
\frac{
|{i: success_i \land L0_i\leq K}|
}{
N_{\text{clean correct}}
}.
]

Đây là cách hợp lý để đưa Sigma-Zero/SparseFool/GSE vào cùng plot với budgeted methods.

Tôi sẽ giữ nguyên concept này.

Chỉ cần sửa conditional robust denominator tương tự.

---

# 21. Checkpoint handling khá tốt

`get_model()` không âm thầm benchmark random initialized model.

Nếu checkpoint không tồn tại:

```python
raise FileNotFoundError(...)
```

Đây là design rất tốt.

Có thêm:

* checkpoint SHA256;
* strict loading;
* optional expected SHA;
* optional min clean accuracy.

Điều này hỗ trợ reproducibility thực sự.

### Nhưng config chưa pin SHA

`paper.yaml` chỉ có:

```yaml
checkpoint: resnet18_cifar10_best.pth
```

Tôi khuyến nghị paper config phải có:

```yaml
checkpoint:
  filename: ...
  sha256: ...
  expected_clean_acc: ...
```

Tên file không đủ để reproducible.

---

# 22. Model architecture phù hợp

ResNet-18 được adapt cho CIFAR:

```python
3x3 conv
stride=1
no maxpool
```

Đúng với protocol.

WideResNet-28-10 cũng đã có implementation.

Đây là lựa chọn tốt cho generalization experiment:

```text
CIFAR10 / ResNet18
CIFAR10 / WRN-28-10
CIFAR100 / WRN-28-10
```

Nếu chỉ benchmark một ResNet18, reviewer rất dễ nói:

> method may be architecture-specific.

---

# 23. Config chưa thực sự chạy đủ dense baselines

Config có kwargs cho:

```yaml
fgsm:
bim:
pgd:
```

nhưng `attacks:` hiện chỉ có:

```yaml
- pgd
- cornersearch
- pgd0
- spgd
- sparse_rs
- sparsefool
- sigma_zero
- gse
- ours
```

FGSM và BIM không nằm trong list.

Trong khi protocol lại liệt kê cả 3 dense references.

Không phải bug lớn, nhưng config và protocol đang lệch nhau.

Tôi sẽ tách:

```yaml
dense_references:
  - fgsm
  - bim
  - pgd

sparse_budgeted:
  ...

minimal:
  ...
```

thay vì một flat `attacks:`.

---

# 24. Dense attacks và Sparse attacks không nên nằm chung ranking

Một lưu ý cho paper.

PGD với:

[
\epsilon=8/255
]

không có cùng threat model với L0 attacks.

Do vậy không nên có table kiểu:

| Attack    | ASR |
| --------- | --: |
| PGD       | ... |
| Sparse-RS | ... |
| Ours      | ... |

rồi xếp hạng trực tiếp.

Dense attacks chỉ nên là:

> reference behavior / dense vulnerability.

Main comparison phải là L0-consistent.

---

# 25. Proposed method hiện có nguy cơ quá nhiều components

Current method gồm:

```text
feature loss
+
spatial interaction
+
CPA / FCSA / HSA / smoothing variants
+
Top-K support selection
+
iterative projected optimization
+
success-first selection
+
support pruning
```

Điều này làm method giàu ý tưởng nhưng **paper story có nguy cơ bị loãng**.

Reviewer sẽ hỏi:

> Contribution thật sự là cái gì?

Tôi sẽ rút core method thành khoảng 3 contribution:

### Contribution A — Feature-aware sparse saliency

[
S_i^{feature}
]

### Contribution B — Collaborative spatial interaction

[
S_i^{coop}
]

### Contribution C — Success-preserving support pruning

và định nghĩa:

[
S_i
===

S_i^{feature}
+
\lambda S_i^{coop}.
]

CPA nên là algorithm chính.

FCSA/HSA nên đưa thành exploratory variants hoặc appendix nếu chưa có mathematical motivation rất mạnh.

---

# 26. Feature guidance hiện chưa trực tiếp influence support score

Một điểm methodology đáng suy nghĩ:

Feature loss có đi vào:

```python
total_loss = CE + feature_loss
loss.backward()
```

do vậy gradient đã bao gồm feature component.

Sau đó interaction score dùng gradient đó.

Điều này ổn.

Nhưng paper cần tránh mô tả như thể đang có:

```text
classification saliency
+
feature saliency
```

hai map riêng biệt, vì implementation hiện tại thực chất là:

[
g =
\nabla_x
(L_{CE}+\lambda L_F)
]

rồi interaction được tính trên (g).

Đó là một formulation khác và nên viết đúng.

---

# 27. Pruning cost có thể rất lớn

Với mỗi successful sample, pruning thử từng active coordinate.

Worst case cho K=64:

```text
up to 64 forward passes/pass
× 5 passes
```

tức ~320 extra forwards/image.

Do đó nếu report:

```text
attack runtime
forward evaluations
```

ours có thể bị ảnh hưởng mạnh.

Nhưng đây không phải điều xấu — nếu L0 giảm đáng kể thì đó là tradeoff đáng nghiên cứu.

Tôi sẽ report hai variants:

```text
Ours
Ours + Prune
```

Ablation rất dễ hiểu.

---

# 28. `paper.yaml` có hyperparameter fairness problem

Current:

```yaml
pgd0:
  steps: 100

spgd:
  steps: 100

sparse_rs:
  n_queries: 10000

sigma_zero:
  steps: 500

gse:
  max_evals: 1000

ours:
  steps: 25
```

Không có gì sai nếu đây là **official recommended settings**.

Nhưng paper phải nói rõ fairness theo loại:

### Option 1 — literature/default settings

Mỗi method chạy config gốc.

### Option 2 — equal compute

Ví dụ:

[
Q = 1k,\ 5k,\ 10k
]

hoặc equal runtime.

### Option 3 — both

Đây là tốt nhất.

Nếu chỉ report default settings, một method 25 gradients và một method 10k queries không nên kết luận đơn giản rằng method này "more efficient" dựa trên ASR alone.

---

# 29. Attack registry design khá đẹp

`ATTACK_REGISTRY` lưu:

```python
name
factory
mode
```

và `create_attack()` filter kwargs dựa trên signature.

Ưu điểm:

* benchmark loop generic;
* thêm baseline đơn giản;
* tránh attack-specific `if/else` everywhere.

Tôi sẽ mở rộng `AttackSpec` thành:

```python
AttackSpec(
    name,
    factory,
    threat_model,
    evaluation_mode,
    gradient_based,
    query_based,
    source,
    official=True/False
)
```

Như vậy benchmark có thể tự validate fairness.

---

# 30. Việc vendoring `third_party/` là tốt nhưng cần quản lý provenance cực chặt

Repo có `THIRD_PARTY.md` và official adapters.

Đây là đúng hướng.

Paper-grade setup nên lưu cho từng baseline:

```text
paper
official repository URL
upstream commit SHA
license
local modifications
wrapper behavior
input normalization
output domain
original hyperparameters
```

Đặc biệt attacks từ các repo khác nhau thường kỳ vọng:

```text
[0,1]
[-1,1]
normalized CIFAR
NHWC
NCHW
```

PGD0 adapter hiện convert:

```python
NCHW Tensor
→ NHWC NumPy
→ official attack
→ NCHW Tensor
```

Đây là nơi rất dễ có silent input-domain mismatch.

---

# 31. Preprocessing consistency cần được kiểm tra

Dataset evaluation transform:

```python
Resize((32, 32))
ToTensor()
```

CIFAR vốn đã 32×32 nên resize dư thừa.

Không nghiêm trọng, nhưng với reproducibility tốt nhất nên bỏ.

Tốt hơn:

```python
transforms.ToTensor()
```

để không có khả năng PIL resize implementation gây subtle difference.

---

# 32. Tests hiện chưa đủ coverage

Hiện có:

```text
test_attacks.py
test_benchmark.py
test_core.py
```

theo tree repo.

Việc tồn tại test suite là tốt.

Nhưng lỗi `BatchMetrics` cho thấy coverage/execution state còn vấn đề.

Tôi đề nghị tối thiểu có các invariant tests sau:

```text
L0(x_adv - x) <= K
x_adv ∈ [0,1]
ASR denominator = clean-correct
conditional_RA + ASR = 100%
minimal ASR curve monotonic in K
ASR@K non-decreasing
L0 projection <= K
exact top-k count = min(K, HW)
all attacks preserve batch/device/dtype
adaptive defense gradients non-zero
feature attack works through defended wrapper
```

Đặc biệt:

[
ASR@K_1 \le ASR@K_2,\quad K_1<K_2
]

nên là automated benchmark sanity check.

---

# 33. Repo chưa có dấu hiệu CI trong tree chính

Từ recursive tree tôi không thấy `.github/workflows/...`.

Với research code có nhiều wrappers từ third-party, CI rất nên có:

```text
ruff / lint
pytest CPU
small synthetic benchmark
L0 contract checks
```

Không cần chạy full CIFAR benchmark.

Chỉ cần 2–4 synthetic samples là bắt được lỗi `BatchMetrics` hiện tại.

---

# 34. Error handling ở benchmark hơi nguy hiểm cho paper runs

Trong scripts:

```python
try:
    ...
except Exception as e:
    result = {"error": str(e)}
```

Ưu điểm: benchmark tiếp tục.

Nhưng với paper experiment, đây là nguy hiểm.

Ví dụ 5/9 attacks fail nhưng script vẫn:

```text
Benchmark Completed!
```

và tạo JSON.

Một downstream table generator có thể vô tình bỏ rows lỗi.

Tôi đề nghị:

```yaml
benchmark:
  fail_fast: true
```

cho paper mode.

Sau benchmark:

```python
if any_errors:
    raise RuntimeError(...)
```

Development mode mới dùng `continue_on_error`.

---

# 35. Current state của defense benchmark chưa đủ để kết luận robustness

Hiện script chạy:

```text
4 preprocessing defenses
× 2 evaluation modes
× 3 sparse attacks
```

Concept tốt.

Nhưng để claim defense robustness, tôi sẽ cần thêm:

```text
Clean accuracy after defense
Oblivious attack
Adaptive/BPDA attack
Attack-specific strongest adaptive settings
EOT nếu defense randomized
AT model
Sparse-AT model
```

Hiện defenses deterministic nên chưa cần EOT.

---

# 36. Một issue conceptual: clean correctness thay đổi dưới defense

`evaluate_batch()` với defended model dùng:

```python
clean_pred = model.evaluate_defended(x)
adv_pred = model.evaluate_defended(x_adv)
```

Điều này nghĩa là ASR defense được condition trên:

> images correctly classified **after defense preprocessing**.

Đó là một metric hợp lệ.

Nhưng khi so sánh defense A vs defense B, denominator có thể khác nhau.

Ví dụ:

```text
Base clean correct: 950
JPEG clean correct: 880
Median clean correct: 820
```

ASR conditional trên từng defense không trực tiếp comparable.

Paper nên báo cả:

```text
Defended Clean Accuracy
Conditional ASR
Full-set Robust Accuracy
```

Repo đã có concept full-set robust accuracy, đó là tốt.

---

# 37. Đánh giá từng phần

| Phần                                |           Đánh giá |
| ----------------------------------- | -----------------: |
| Repository architecture             |           **8/10** |
| Research protocol                   |         **8.5/10** |
| Reproducibility design              |           **8/10** |
| Data sampling                       |           **8/10** |
| Model handling                      |           **8/10** |
| Baseline organization               |         **7.5/10** |
| Proposed-method concept             |       **7.5–8/10** |
| Proposed implementation correctness |         **5.5/10** |
| Metric correctness                  |           **4/10** |
| Defense methodology                 |           **7/10** |
| Defense completeness                |         **4.5/10** |
| Tests / reliability                 |           **5/10** |
| Paper readiness                     | **~5/10 hiện tại** |

Sau khi sửa các P0/P1 issues, tôi nghĩ codebase có thể lên **7.5–8/10 paper readiness** khá nhanh.

---

# 38. Priority fixes

Tôi sẽ xử lý theo thứ tự này:

1. **Sửa `BatchMetrics` ngay.** Đây là runtime blocker.
2. **Sửa conditional robust accuracy denominator.**
3. **Sửa FeatureExtractorAdapter để hoạt động qua `DefendedModelAdapter`.**
4. **Sửa FCSA**, vì synergy hiện identically zero.
5. **Loại circular neighborhood trong CPA.**
6. **Refactor query/forward/backward counting** thành measurement thực.
7. **Thêm benchmark contract tests.**
8. **Thêm CI.**
9. **Tăng benchmark samples lên ≥1000 cho intermediate và ideally full test cho final.**
10. **Implement adversarial training defense.**
11. **Pin checkpoint SHA trong config.**
12. **Chuẩn hóa third-party input/output and provenance.**

---

# 39. Proposed experimental architecture tôi khuyến nghị

Sau khi sửa correctness, tôi sẽ tổ chức experiments thành 5 blocks.

### A. Main sparse benchmark

```text
Models:
ResNet18
WRN-28-10

K:
1,2,4,8,16,32,64

Attacks:
CornerSearch
PGD0
Sparse-PGD
Sparse-RS
SparseFool
Sigma-Zero
GSE
Ours
```

Report:

```text
ASR@K
Robust Accuracy
Median successful L0
L2
Linf
runtime
model evaluations
```

---

### B. Minimal-L0 comparison

```text
SparseFool
Sigma-Zero
GSE
Ours + pruning
```

Report:

```text
median L0
mean L0
ASR threshold curve
CDF of required L0
```

Một plot rất mạnh sẽ là:

[
P(L_0^\star\le K)
]

theo K.

---

### C. Ablation

```text
CE only
+ Feature
+ CPA
+ Feature + CPA
+ Pruning
Full
```

Đây tốt hơn một ablation quá lớn với FCSA/HSA.

---

### D. Efficiency

Report:

```text
ASR vs queries
ASR vs runtime
ASR vs gradients
L0 vs runtime
```

---

### E. Defense

```text
None
Blur
Median
JPEG
TV
PGD-AT
Sparse-AT
```

với:

```text
oblivious
adaptive/BPDA
```

---

# 40. Main concern về novelty

Kiến trúc proposed hiện có tiềm năng, nhưng novelty không nên được đặt vào:

> "Top-K gradient + feature loss"

vì riêng từng thành phần này khá gần những motif đã quen thuộc trong adversarial attack literature.

Phần đáng phát triển nhất theo tôi là:

[
\boxed{
\text{Collaborative pixel selection under strict spatial }L_0
}
]

tức là **pixel không được chọn độc lập**, mà score phụ thuộc interaction của local gradient vectors / receptive-field collaboration.

Nếu formalize tốt CPA thành một principled optimization surrogate, đây sẽ là story mạnh hơn nhiều.

Ví dụ objective:

[
\max_{S: |S|\le K}
\left[
\sum_{i\in S}u_i
+
\lambda
\sum_{(i,j)\in E_S}w_{ij}
\right]
]

trong đó:

[
u_i = |g_i|,
]

và

[
w_{ij}
======

\max(0,\cos(g_i,g_j))
|g_i||g_j|.
]

Khi đó attack trở thành một dạng:

> **budgeted cooperative support selection**.

CPA hiện tại có thể được diễn giải như một approximation hiệu quả cho objective này.

Đó là hướng mà tôi đánh giá có chất lượng paper cao hơn việc đưa 4 interaction modes song song.

---

# Kết luận

**AA_2 có foundation tốt hơn một prototype thông thường.** Repository architecture, protocol, deterministic sampling, checkpoint validation, separation giữa official baselines / proposed method và adaptive defense thinking đều khá tốt.

Nhưng **tôi chưa khuyến nghị sử dụng bất kỳ benchmark number nào từ version hiện tại làm số liệu paper**, chủ yếu vì:

* `BatchMetrics` đang có lỗi interface nghiêm trọng;
* conditional robust accuracy tính sai;
* proposed attack có FCSA degenerating to zero synergy;
* CPA có wrap-around spatial interaction;
* proposed method có khả năng không chạy qua defense wrapper;
* compute/query accounting chưa fair;
* adversarial-training defense chưa tồn tại;
* paper config chỉ dùng 100 samples.

Nếu sửa nhóm P0 trước, tôi đánh giá **repo hoàn toàn có thể trở thành nền tảng khá mạnh để phát triển paper Sparse AA**, và phần nên tập trung nhất là **collaborative support selection + feature guidance + success-preserving pruning**, thay vì mở rộng thêm nhiều heuristic interaction modes.
