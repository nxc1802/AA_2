Mình đã đọc cấu trúc repository, protocol, benchmark loop, data/model pipeline, proposed attack, defense, external adapters, tests và lịch sử commit gần nhất. Kết luận chính: **AA_2 có kiến trúc nghiên cứu gọn và hướng paper tốt, nhưng snapshot `main` hiện tại chưa đủ an toàn để dùng sinh bảng kết quả paper**. Có vài lỗi thiết kế có thể làm benchmark sai về mặt khoa học dù chương trình vẫn chạy.

Repo hiện tại là Python research package với `src/aa`, benchmark CLI, protocol riêng và adapters cho các sparse attack chính.  Đáng chú ý, commit mới nhất là một refactor cực lớn từ codebase cũ sang package tối giản; so với commit ngay trước đó, nhiều module CPA/FCSA/HSA, benchmark infrastructure và test validation đã bị xóa/thay thế. Vì vậy đây không chỉ là cleanup mà thực chất là một **rewrite nghiên cứu** cần re-validation.

## 1. Đánh giá tổng thể

Nếu chấm theo mục tiêu làm một paper Sparse Adversarial Attack:

| Thành phần                     | Đánh giá |
| ------------------------------ | -------: |
| Organization / code structure  | **8/10** |
| Threat-model definition        | **8/10** |
| Reproducibility design         | **5/10** |
| Baseline coverage              | **8/10** |
| Baseline fairness hiện tại     | **4/10** |
| Proposed-method implementation | **5/10** |
| Defense evaluation             | **3/10** |
| Metrics                        | **6/10** |
| Tests                          | **4/10** |
| Paper-readiness                | **4/10** |

Điểm mạnh nhất là project đã hiểu đúng rằng paper attack cần một **protocol làm single source of truth**, cùng sample set, spatial (L_0), conditional ASR, distortion và computational cost. Protocol định nghĩa khá tốt spatial (L_0), tập (K={1,2,4,8,16,32,64}), clean-correct conditional ASR và danh sách baseline.

Nhưng implementation hiện chưa tuân thủ hoàn toàn chính protocol này.

---

# 2. Các vấn đề P0 — nên sửa trước khi chạy paper benchmark

## P0.1 — Sample benchmark **không stratified như protocol yêu cầu**

Protocol ghi:

> All attacks compared within the same experiment must use exactly the same sample indices. Sampling must be deterministic, class-stratified...

Nhưng implementation:

```python
hf_ds = load_dataset(
    HF_REPO_ID,
    name=ds_name,
    split=f"test[:{num_samples}]"
)
```

Tức chỉ lấy **N phần tử đầu tiên của test set**.

Không hề stratify, cũng không sử dụng `seed`.

Đây là lỗi khoa học khá nghiêm trọng.

Ví dụ `samples: 100` hiện tại trong `paper.yaml`:

100 sample đầu có thể có distribution không cân bằng hoặc vô tình dễ/khó hơn toàn dataset.

### Nên đổi

Tạo một manifest cố định:

```text
benchmark_indices/
    cifar10_test_seed42_n1000.json
```

với đúng số sample/class.

Ví dụ N=1000:

```text
100 images/class × 10 classes
```

Sau đó **mọi attack đều dùng chính xác manifest này**.

---

# 3. P0.2 — Checkpoint không tồn tại → code vẫn chạy model random

Đây là lỗi nguy hiểm nhất trong toàn repo.

Trong `get_model()`:

```python
resolved_ckpt = find_existing_checkpoint(checkpoint_path)

if resolved_ckpt is not None:
    ...
    model.load_state_dict(...)
```

Nếu checkpoint không tìm thấy, code **không raise exception** mà tiếp tục trả về model random initialized.

Trong config:

```yaml
checkpoint: resnet18_cifar10_best.pth
```

Nếu download HF lỗi, token lỗi, filename sai hoặc offline, benchmark vẫn chạy bình thường.

Kết quả có thể kiểu:

```text
Clean Acc ≈ 10%
ASR = ...
```

và vẫn được ghi JSON.

### Đây phải là hard failure

Nên:

```python
if resolved_ckpt is None:
    raise FileNotFoundError(...)
```

Paper mode còn nên kiểm tra:

```text
expected SHA256
actual SHA256
architecture
dataset
clean test accuracy
```

Ví dụ yêu cầu:

```text
ResNet18 CIFAR-10 checkpoint
expected clean acc >= 93%
```

nếu thấp hơn ngưỡng thì benchmark abort.

---

# 4. P0.3 — `paper.yaml` chạy dense PGD theo từng K là không có nghĩa

Runner làm:

```python
for atk_name in attack_names:
    for k in k_values:
        attack = create_attack(atk_name, model=model, k=k)
```

Nhưng registry lọc kwargs theo constructor. PGD không có parameter `k`, nên:

```python
create_attack("pgd", k=1)
create_attack("pgd", k=2)
...
create_attack("pgd", k=64)
```

thực chất đều tạo cùng một:

```text
PGD L∞ eps=8/255
```

Registry thực hiện silent filtering này ở đây.

Vì PGD có random start, mỗi "K" thậm chí có thể cho **kết quả khác nhau do randomness**, mặc dù K hoàn toàn không tồn tại trong attack.

Khi plot:

```text
PGD ASR@1
PGD ASR@2
PGD ASR@4
...
```

sẽ cực kỳ misleading.

### Thiết kế đúng

Registry đã có `mode`:

```python
dense
budget
minimal
```

hãy dùng chính nó.

* Dense: chạy **1 lần** với (\epsilon).
* Budget attack: sweep K.
* Minimal-support: chạy 1 lần rồi derive ASR@K từ achieved L0, hoặc sweep parameter chỉ khi thuật toán thực sự định nghĩa K.

---

# 5. P0.4 — Proposed method chạy trong benchmark **không phải Full Method trong document**

Document định nghĩa:

```text
A3 Full Method:
feature_guidance=True
interaction=True
pruning=True
```

Nhưng constructor:

```python
feature_guidance=True
interaction=False
pruning=True
```

Runner chỉ gọi:

```python
create_attack(atk_name, model=model, k=k)
```

Do đó `"ours"` trong paper benchmark hiện tại là:

```text
Feature = ON
Interaction = OFF
Pruning = ON
```

chứ **không phải A3 Full Method**.

Đây là mismatch trực tiếp giữa Method section và Experiment section.

---

# 6. P0.5 — Proposed method sau refactor không còn đúng bản proposed trước đó

Lịch sử commit rất đáng chú ý.

Ngay trước refactor có commit:

```text
fix(proposed): Implement mathematical formulations for
CPA, FCSA, HSA ...
```

Sau đó commit mới nhất:

```text
refactor: consolidate codebase into minimal aa research package
```

Refactor đã xóa:

```text
src/attacks/proposed/cpa.py
src/attacks/proposed/fcsa.py
src/attacks/proposed/fmsa.py
src/attacks/proposed/hsa.py
```

và gom tất cả thành một `SparseFeatureAttack`.

CPA cũ thực sự tính directional gradient cooperation:

[
I(i)=|g_i|*1+
\lambda\sum*{j\in N(i)}
\operatorname{ReLU}(\cos(g_i,g_j))
|g_j|_1
]

Implementation cũ có normalized gradient, cosine alignment giữa neighboring pixels và weighted neighbor magnitude.

Implementation mới khi `interaction=True` chỉ làm:

```python
grad_mag = grad_mag + 0.5 * avg_pool2d(
    grad_mag,
    kernel_size=3,
    stride=1,
    padding=1,
)
```

Đây chỉ là **local smoothing của gradient magnitude**.

Nó không còn:

* directional alignment,
* cosine interaction,
* cooperative pixels formulation.

Do đó nếu paper vẫn claim CPA/FCSA/HSA hoặc "pixel cooperation", implementation hiện tại không support claim đó.

Đây là regression về **research semantics**, không phải code style.

---

# 7. P0.6 — Defense benchmark hiện không đánh giá Sparse Attack

Defense runner hard-code:

```python
atk = create_attack("pgd", model=defended_model, k=16)
```

Như trên, `k=16` bị PGD bỏ qua.

Nghĩa là defense benchmark thực chất đang đo:

[
L_\infty\text{-PGD}
]

trên preprocessing defense.

Trong khi research question của project là:

> How does sparse robustness change under preprocessing and adversarial-training defenses?

Hai câu hỏi khác nhau.

Defense table của Sparse AA ít nhất phải chứa:

```text
Attack:
CornerSearch
Sparse-RS
Sparse-PGD
Sigma-Zero/GSE
Ours

Defense:
None
Blur
Median
JPEG
TVM
Adversarially trained model
```

và attack phải adaptive.

---

# 8. P0.7 — Adversarial training gần như chưa tồn tại

Protocol nói nghiên cứu cả:

```text
preprocessing
adversarial training
```

Nhưng snapshot hiện tại chỉ có preprocessing defense trong `defenses.py`.

Không thấy pipeline:

```text
PGD adversarial training
TRADES
MART
sparse adversarial training
checkpoint robust
```

Do đó phần defense hiện tại mới khoảng **30–40% phạm vi project**.

---

# 9. P0.8 — Efficiency accounting chưa đủ tin cậy

Protocol muốn report:

```text
wall-clock runtime
forward evaluations
backward evaluations
queries
```

Generic benchmark cộng metadata từ `AttackOutput`:

```python
total_forward += output.forward_evals
total_backward += output.backward_evals
total_queries += output.queries
```

Nhưng các adapters tự estimate khác nhau.

PGD0:

```python
forward_evals=self.steps
backward_evals=self.steps
```

SparseFool:

```python
forward_evals=total_loops
backward_evals=total_loops
```

Nhưng một "loop" của upstream algorithm không nhất thiết tương ứng chính xác 1 forward + 1 backward.

Ngoài ra benchmark tự gọi:

```python
model(x)
model(x_adv)
```

để evaluate success nhưng không cộng vào accounting.

Nếu metric muốn là **attack cost**, bỏ evaluation forwards là hợp lý, nhưng phải ghi rõ.

Hiện `forward_evals` và `queries` đang mang semantics khác nhau giữa attack.

### Tốt hơn nhiều

Dùng model wrapper:

```python
class CountingModel(nn.Module):
    forward_calls
    samples_evaluated
```

và gradient instrumentation riêng.

Đặc biệt query-based attacks nên report:

[
\text{queries / image}
]

không chỉ tổng queries.

---

# 10. Metrics: phần tốt và phần cần sửa

## Spatial L0

Phần này làm tốt:

```python
channel_max = delta.abs().max(dim=1).values
l0 = (channel_max > eps).flatten(1).sum(dim=1)
```

Đây đúng với spatial (L_0):

> pixel được tính modified nếu ít nhất một channel thay đổi.

Projection cũng đúng conceptual:

```python
spatial_mag = torch.norm(delta, p=2, dim=1)
topk(...)
```

rồi giữ nguyên cả 3 channel tại K pixel.

Đây là một phần core tốt của repo.

---

## ASR

Implementation:

```python
success = clean_correct & adv_pred.ne(y)
```

và:

```python
ASR = success_count / clean_correct_count
```

Khớp protocol.

Đây là đúng.

---

## Robust Accuracy

Implementation hiện tính:

```python
robust_acc =
    (clean_correct_count - succ_count) / total_samples
```

Nó tương đương:

```text
clean-correct AND still-correct-after-attack
```

chứ không phải trực tiếp:

```python
adv_pred.eq(y).mean()
```

Trong phần lớn attack benchmark hai giá trị gần nhau, nhưng semantic không hoàn toàn giống nhau nếu một sample clean-incorrect trở thành correct sau perturbation.

Nên report hai metric rõ:

```text
Robust Accuracy Full Set
Conditional Robust Accuracy
```

---

# 11. LPIPS được khai báo nhưng thực tế không chạy

`evaluate_attack()` nhận:

```python
lpips_fn=None
```

nhưng cuối cùng gọi:

```python
lpips_per=None
```

hard-code.

Trong khi `metrics.py` đã có:

```python
compute_per_sample_lpips(...)
```

Do đó protocol claim PSNR/SSIM/LPIPS nhưng benchmark hiện chỉ thực sự report PSNR/SSIM.

Đây là straightforward bug/incomplete integration.

---

# 12. Proposed Method — phân tích thuật toán

Current method về bản chất là:

[
\mathcal L =
CE(f(x'),y)+
\lambda
|\phi(x')-\phi(x)|_2^2
]

sau đó:

1. gradient theo input;
2. score pixel bằng tổng absolute gradient channels;
3. tùy chọn local smoothing;
4. lấy top-K;
5. sign-gradient update;
6. exact L0 projection;
7. success-first candidate selection;
8. early stop;
9. greedy support pruning.

Implementation tương ứng khá sát mô tả basic SFA.

### Điểm tốt

**Success-first selection tốt.**

Ưu tiên:

```text
successful > unsuccessful
lower L0 > higher L0
higher CE if same L0
```

đây là logic hợp lý cho sparse attack.

**Freeze fooled samples tốt.**

```python
fooled_mask
```

giúp giảm gradient computation không cần thiết.

**Exact top-K projection tốt.**

Không để perturbation vô tình vượt K.

**Greedy pruning có giá trị thực tiễn.**

Sau khi fool thành công, thử remove từng active pixel giúp giảm support.

---

# 13. Nhưng proposed method có vài vấn đề nghiên cứu

## 13.1 Pruning là order-dependent

Code:

```python
for coord in active_coords:
    remove pixel
    if still fooled:
        keep removal
```

Nếu support là `{a,b,c}`, có thể:

```text
remove a first -> fail
remove b -> success
...
```

nhưng sau khi b bị remove, lúc này a có thể trở thành removable.

Code không quay lại test a lần hai.

Do đó output chỉ là **1-pass greedy minimal**, không phải support-minimal.

Nên ít nhất iterative until convergence:

```text
repeat:
    attempt removal of all active pixels
until no removal succeeds
```

Tốt hơn nữa, prune theo contribution score nhỏ → lớn.

---

## 13.2 Sau khi prune không re-optimize remaining pixels

Một sparse attack mạnh hơn có thể:

```text
remove pixel
→ optimize values on remaining support
→ test success
```

Current pruning chỉ set 0 và test.

Vì vậy nhiều support redundant có thể không được loại do remaining pixels chưa reoptimized.

---

## 13.3 Feature loss scale chưa normalized

```python
feat_l = feat_diff.pow(2).flatten(1).sum(dim=1)
```

Đây là `sum`, phụ thuộc trực tiếp:

```text
layer spatial resolution
number of channels
architecture
```

Trong khi CE ~ O(1).

Feature loss có thể rất lớn.

`feature_weight=1.0` vì thế khó transfer ResNet18 → WRN.

Nên dùng:

[
\frac{1}{d}|\phi(x')-\phi(x)|_2^2
]

hoặc normalized representation:

[
1-\cos(\phi(x'),\phi(x))
]

và report λ ablation.

---

## 13.4 Layer handling quá cứng

Feature extractor:

```python
elif hasattr(model, "layer4"):
...
elif hasattr(model, "features"):
```

WRN custom trong repo lại có:

```text
block1
block2
block3
```

Không có `layer4` hoặc `features`.

Vậy với WRN:

```text
feature_adapter.extracted_features = None
```

và SFA âm thầm degrade về CE attack.

Đây là một lỗi lớn cho cross-architecture generalization.

Không nên silent fallback.

Nên:

```python
if feature_guidance and layer is None:
    raise ValueError(...)
```

---

# 14. Dense baselines: ổn cho reference nhưng cần phân loại rõ

FGSM/BIM/PGD implementation cơ bản đúng (L_\infty).

Nhưng chúng **không phải sparse threat-model competitors**.

Trong paper nên chia:

### Dense attack references

```text
FGSM
BIM
PGD-L∞
```

dùng để trả lời:

> model có vulnerability thông thường không?

### Sparse competitors

```text
CornerSearch
PGD0
Sparse-PGD
Sparse-RS
SparseFool
Sigma-Zero
GSE
Ours
```

Dense PGD không nên xuất hiện cùng bảng ASR@K như thể cùng constraint.

---

# 15. Minimal attacks cần protocol riêng

Registry có:

```text
budget
minimal
dense
```

đây là ý tưởng rất tốt.

Nhưng runner chưa sử dụng semantics đó.

Ví dụ SparseFool constructor nhận `k`, nhưng code hoàn toàn không dùng `self.k` trong attack.

Vậy sweep:

```text
SparseFool K=1
SparseFool K=2
...
```

không có nghĩa.

### Cách benchmark minimal-support attack đúng hơn

Chạy attack để thu:

[
L_0^{(i)}
]

rồi tính:

[
ASR@K =
\frac{
#{i:
success_i\land L_0^{(i)}\le K
}
}{
# clean\ correct
}
]

Như vậy một lần chạy có thể derive toàn curve K.

Đây sẽ là benchmark sạch hơn rất nhiều.

---

# 16. Defense implementation

## Gaussian blur

Differentiable, nên adaptive attack có thể backprop trực tiếp.

Tốt.

## Median/JPEG/TVM

Repo sử dụng BPDA straight-through:

```python
backward:
    return grad_output
```

Đây là baseline adaptive evaluation hợp lý.

Điểm này mình đánh giá tốt về awareness đối với **gradient masking**.

Tức tác giả không chỉ attack preprocess một cách oblivious.

---

# 17. Tuy nhiên `oblivious` adapter có semantic đáng ngờ

Trong:

```python
if mode == "adaptive":
    ...
else:
    return self.model(x)
```

Ở `oblivious` mode defense bị bỏ hoàn toàn.

Nếu intention là:

> attacker không biết defense nhưng evaluator vẫn đánh giá defended prediction

thì architecture đúng thường phải tách:

```text
attack_model = base_model
evaluation_model = defense + base_model
```

Chứ không phải defended model forward bỏ defense.

Hiện script chỉ dùng adaptive nên bug này chưa tác động main path, nhưng abstraction nên sửa.

---

# 18. TVM hiện tại chưa thực sự là chuẩn Total Variation Minimization

Implementation tính:

```python
diff_h = abs(...)
diff_w = abs(...)
...
x_def -= step_size * (...)
```

Nó giống custom iterative smoothing dựa trên finite differences hơn là một TV denoising optimizer được xác định rõ.

Nếu paper gọi đây là "TVM", reviewer có thể hỏi formulation/reference.

Nên:

* hoặc dùng implementation chuẩn;
* hoặc gọi đúng là `"TV smoothing"`;
* mô tả objective và optimizer chính xác.

---

# 19. Tests hiện tại quá nhẹ cho một benchmark paper

`test_attacks.py` chỉ test:

```text
shape
[0,1] bounds
L0 <= K cho ours
```

và chỉ:

```text
fgsm
pgd
ours
```

Nhiều validation tests từ codebase cũ đã bị xóa trong refactor.

`test_benchmark.py` chủ yếu kiểm tra clean accuracy arithmetic trên dummy model.

Đối với paper benchmark, cần tests mạnh hơn.

Mình sẽ coi những test sau là bắt buộc:

```text
test_all_budget_attacks_respect_k

test_all_outputs_in_pixel_domain

test_same_indices_across_attacks

test_stratified_sampler

test_missing_checkpoint_fails

test_checkpoint_sha

test_dense_attacks_not_swept_by_k

test_minimal_attacks_not_swept_by_k

test_conditional_asr_definition

test_full_robust_accuracy

test_proposed_full_config_matches_doc

test_feature_layer_exists_for_each_model

test_bpda_gradient_nonzero

test_defense_actually_applied

test_query_counter

test_lpips_pipeline

test_external_adapter_smoke_each_attack
```

---

# 20. Reproducibility

Protocol có mục tiêu đúng: save dataset, architecture, seed, validation/test accuracy, checkpoint SHA256, git commit.

Runner cũng save:

```python
"reproducibility": get_git_reproducibility_info()
```

Đây là hướng rất tốt.

Nhưng paper output nên có thêm:

```text
Python version
PyTorch version
CUDA version
GPU
dataset fingerprint
sample-index SHA256
checkpoint SHA256
third-party baseline commit hash
attack hyperparameters
wall-clock timestamp
git dirty status
```

Đặc biệt **third-party commit hash** quan trọng vì repo vendor source code.

---

# 21. README có bug portability nhỏ nhưng rõ

README đang link:

```text
file:///Volumes/WorkSpace/Project/...
```

Đây là local path của máy developer.

Trên GitHub link sẽ không portable.

Nên dùng:

```markdown
[Experimental Protocol](docs/protocol.md)
```

Tương tự cho các docs khác.

Không ảnh hưởng science, nhưng làm repo chưa polished.

---

# 22. Config hiện tại chưa đủ cho reproducible paper

`paper.yaml` chỉ khoảng:

```yaml
seed
dataset
samples
batch_size
model
checkpoint
k_values
attacks
```

Attack hyperparameters vẫn nằm trong constructor defaults.

Điều này nguy hiểm.

Nếu sửa:

```python
SparseFeatureAttack.steps = 50
```

6 tháng sau chạy cùng `paper.yaml` sẽ ra experiment khác.

### Paper config nên tự-contained

Ví dụ:

```yaml
attacks:
  pgd:
    eps: 0.0313725
    alpha: 0.007843
    steps: 20
    restarts: 5

  sparse_rs:
    queries: 10000

  ours:
    steps: 50
    alpha: 0.015686
    feature_weight: 0.1
    feature_guidance: true
    interaction: true
    pruning: true
    layer: layer4
```

**Không để paper experiments phụ thuộc constructor defaults.**

---

# 23. Một vấn đề experimental-design khác: chỉ 100 samples

Current config:

```yaml
samples: 100
```

100 sample phù hợp **smoke benchmark**, không phù hợp main paper table.

Ví dụ ASR 50% trên n=100 có standard error khoảng:

[
\sqrt{\frac{0.5(0.5)}{100}}=5%
]

95% CI gần ±10 percentage points.

Quá lớn để claim:

```text
ours 62%
baseline 58%
```

### Khuyến nghị

Development:

```text
100 images
```

Ablation:

```text
1,000 images
```

Main benchmark:

```text
entire 10,000 CIFAR-10 test
```

hoặc ít nhất 1,000–5,000 nếu attacks quá đắt, với confidence intervals.

---

# 24. Kiến trúc project mình đề xuất

Giữ tinh thần minimal package hiện tại nhưng tách rõ hơn:

```text
src/aa/
├── attacks/
│   ├── dense/
│   ├── sparse_budget/
│   ├── sparse_minimal/
│   ├── proposed/
│   └── adapters/
│
├── benchmark/
│   ├── runner.py
│   ├── protocols.py
│   ├── accounting.py
│   └── result_schema.py
│
├── datasets/
│   ├── loader.py
│   └── benchmark_manifest.py
│
├── defenses/
│   ├── preprocessing/
│   ├── bpda.py
│   └── robust_models.py
│
├── metrics/
│   ├── attack.py
│   ├── distortion.py
│   └── perceptual.py
│
├── models/
│   ├── factory.py
│   └── checkpoints.py
│
└── reproducibility/
    ├── fingerprint.py
    └── provenance.py
```

Không cần quay lại codebase cũ 50 files/module quá phức tạp.

**Minimal refactor hiện tại là đúng hướng**, nhưng đã simplify quá mạnh đến mức mất một số scientific invariants.

---

# 25. Proposed method nên tách riêng như thế nào

Vì proposed method còn thay đổi, mình đặc biệt khuyến nghị không nhúng methodology assumptions trực tiếp vào generic infrastructure.

Ví dụ:

```text
src/aa/attacks/proposed/
├── attack.py
├── feature_objective.py
├── support_score.py
├── optimizer.py
├── pruning.py
└── configs.py
```

Generic benchmark chỉ cần biết:

```python
AttackOutput
```

Như vậy bạn có thể thay:

```text
SFA v1
CPA
FCSA
Hybrid SFA
frequency-guided
interaction-guided
```

mà không ảnh hưởng benchmark framework.

Đây đặc biệt hợp với mục tiêu nghiên cứu hiện tại vì proposed method còn có khả năng thay đổi.

---

# 26. Protocol benchmark mình khuyên dùng

Main Sparse Table:

| Attack       | ASR@1 | @2 | @4 | @8 | @16 | @32 | @64 | Median L0 | Queries | Time |
| ------------ | ----: | -: | -: | -: | --: | --: | --: | --------: | ------: | ---: |
| CornerSearch |       |    |    |    |     |     |     |           |         |      |
| PGD0         |       |    |    |    |     |     |     |           |         |      |
| Sparse-PGD   |       |    |    |    |     |     |     |           |         |      |
| Sparse-RS    |       |    |    |    |     |     |     |           |         |      |
| SparseFool   |       |    |    |    |     |     |     |           |         |      |
| Sigma-Zero   |       |    |    |    |     |     |     |           |         |      |
| GSE          |       |    |    |    |     |     |     |           |         |      |
| **Ours**     |       |    |    |    |     |     |     |           |         |      |

Dense reference table riêng:

| Attack | Constraint       | ASR | L2 | Linf |
| ------ | ---------------- | --: | -: | ---: |
| FGSM   | (L_\infty,8/255) |     |    |      |
| BIM    | (L_\infty,8/255) |     |    |      |
| PGD    | (L_\infty,8/255) |     |    |      |

Không trộn hai threat model.

---

# 27. Defense table nên thiết kế lại

Ít nhất:

| Defense   | Clean Acc | Sparse-RS RA@16 | PGD0 RA@16 | Ours RA@16 |
| --------- | --------: | --------------: | ---------: | ---------: |
| None      |           |                 |            |            |
| Gaussian  |           |                 |            |            |
| Median    |           |                 |            |            |
| JPEG      |           |                 |            |            |
| TV        |           |                 |            |            |
| PGD-AT    |           |                 |            |            |
| TRADES    |           |                 |            |            |
| Sparse-AT |           |                 |            |            |

Và mỗi preprocessing defense:

```text
Oblivious attack
Adaptive attack/BPDA
```

Nếu adaptive phá được defense nhưng oblivious không phá được, đó là dấu hiệu classic gradient masking.

---

# 28. Ablation proposed method

Document hiện đã có idea A0–A3 khá tốt.

Mình sẽ mở rộng thành:

```text
B0 CE + top-K gradient
B1 + feature disruption
B2 + interaction
B3 + success-first
B4 + pruning
B5 + iterative pruning
Full
```

Và hyperparameter:

```text
feature layer
λ_feature
interaction weight
steps
alpha
pruning rounds
```

Metric không chỉ ASR.

Nên xem:

```text
ASR@K
median achieved L0
queries
time
PSNR
```

Một component tăng ASR nhưng tăng 10× queries có thể không thực sự tốt.

---

# 29. Priority roadmap

Mình sẽ sửa theo thứ tự này:

1. **Hard fail khi checkpoint missing.**
2. **Fix deterministic stratified evaluation manifest.**
3. **Runner aware `dense/budget/minimal`.**
4. **Fix `"ours"` để thực sự chạy Full Method.**
5. **Quyết định proposed algorithm:** giữ simplified SFA hay phục hồi CPA/FCSA/HSA formulation.
6. **Restore proper query/evaluation accounting.**
7. **Integrate LPIPS.**
8. **Defense benchmark bằng sparse attacks.**
9. **Thêm robust/adversarial-training checkpoints.**
10. **Restore scientific contract tests.**
11. Chuyển toàn attack hyperparameters vào YAML.
12. Tăng sample size.
13. Thêm CI + experiment validation gate.
14. Sau đó mới chạy full benchmark/paper tables.

---

# Kết luận

**Phần infrastructure của AA_2 có nền tảng tốt hơn rất nhiều so với một codebase research prototype thông thường.** Những thứ đáng giữ là:

* protocol riêng;
* registry;
* generic `AttackOutput`;
* spatial (L_0) implementation;
* unified evaluator;
* official third-party adapters;
* BPDA awareness;
* checkpoint/provenance concept;
* tách proposed method khỏi baseline;
* minimal `src/aa` package.

Nhưng **commit refactor mới nhất đã optimize quá mạnh cho sự gọn gàng và làm mất một phần rigor của version trước**. Đặc biệt, hiện có 5 vấn đề đủ sức làm invalid hoặc misleading kết quả paper: **random model fallback, non-stratified subset, K sweep sai cho dense/minimal attacks, proposed config không khớp Full Method, và defense benchmark đang dùng PGD-L∞ thay vì sparse attacks.**

Vì vậy mình sẽ coi trạng thái hiện tại là **“research framework prototype / pre-benchmark”**, chưa phải **“paper execution ready”**.

Điểm đáng quan tâm nhất nữa là proposed method: **`SparseFeatureAttack` hiện tại không còn tương đương với hệ CPA/FCSA/HSA mà repository vừa implement trước refactor**. Nếu hướng paper ban đầu của bạn dựa vào pixel cooperation / feature-aware collaborative sparse optimization, mình khuyên phục hồi formulation mạnh đó vào kiến trúc minimal mới thay vì dùng local average-pooling interaction hiện tại.
