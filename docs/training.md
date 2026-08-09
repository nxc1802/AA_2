Có. Dựa trên cấu trúc hiện tại của `AA_2`, mình đề xuất bổ sung training theo hướng **tách hoàn toàn training engine khỏi benchmark engine**, nhưng dùng chung `data.py`, `models.py`, seed/checkpoint/reproducibility utilities. Hiện `scripts/` mới chỉ có `attack_benchmark.py` và `defense_benchmark.py`, nên training là một subsystem còn thiếu.

Mục tiêu cuối cùng nên đạt pipeline:

```text
Dataset
   ↓
Clean Training
   ↓
Clean Checkpoint
   ├─────────────→ Attack Benchmark
   │
   ↓
Adversarial Training
   ↓
Robust Checkpoint
   ├─────────────→ Sparse Attack Benchmark
   └─────────────→ Defense Benchmark
```

# 1. Kiến trúc sau khi bổ sung

Mình đề xuất repo thành:

```text
AA_2/
├── configs/
│   ├── benchmark/
│   │   ├── smoke.yaml
│   │   ├── cifar10_resnet18.yaml
│   │   └── cifar10_wrn.yaml
│   │
│   ├── train/
│   │   ├── clean_cifar10_resnet18.yaml
│   │   ├── clean_cifar10_wrn28_10.yaml
│   │   ├── clean_cifar100_resnet18.yaml
│   │   └── clean_cifar100_wrn28_10.yaml
│   │
│   └── adv_train/
│       ├── pgd_at_cifar10_resnet18.yaml
│       ├── spgd_at_cifar10_resnet18.yaml
│       └── sfa_at_cifar10_resnet18.yaml
│
├── scripts/
│   ├── train_clean.py                # NEW
│   ├── train_adversarial.py          # NEW
│   ├── evaluate_checkpoint.py        # NEW
│   ├── attack_benchmark.py
│   └── defense_benchmark.py
│
├── src/aa/
│   ├── training/                     # NEW
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── adversarial.py
│   │   ├── optim.py
│   │   ├── checkpoint.py
│   │   └── history.py
│   │
│   ├── attacks/
│   ├── benchmark.py
│   ├── data.py
│   ├── models.py
│   ├── defenses.py
│   ├── metrics.py
│   └── utils.py
│
├── tests/
│   ├── test_training.py              # NEW
│   ├── test_adv_training.py          # NEW
│   ├── test_checkpoint.py            # NEW
│   └── ...
│
└── result/
    ├── checkpoints/
    ├── training_logs/
    └── benchmark/
```

Điểm quan trọng: **không nhét toàn bộ logic vào `train_clean.py`**. Script chỉ đọc config và gọi training engine.

---

# 2. Phase 0 — sửa abstraction model/dataset trước

Đây nên là việc đầu tiên.

Hiện `get_model()` hỗ trợ ResNet-18, ResNet-50 và WRN-28-10, nhưng mặc định `num_classes=10`, còn tên checkpoint mặc định cũng gắn `_cifar10`.

Trong khi `data.py` đã cho phép truyền `dataset_name`, và protocol dự kiến CIFAR-100 sau CIFAR-10.

## 2.1 Dataset metadata

Nên thêm:

```python
DATASET_SPECS = {
    "cifar10": {
        "num_classes": 10,
        "image_size": 32,
        "channels": 3,
    },
    "cifar100": {
        "num_classes": 100,
        "image_size": 32,
        "channels": 3,
    },
}
```

Và helper:

```python
get_dataset_spec("cifar10")
```

Sau đó model luôn được khởi tạo dựa vào dataset:

```python
num_classes = dataset_spec["num_classes"]

model = get_model(
    model_name="resnet18",
    num_classes=num_classes
)
```

### Acceptance criterion

Không được tồn tại logic:

```python
if dataset == "cifar100":
    num_classes = 100
```

rải rác ở nhiều script.

Chỉ có **một source of truth**.

---

# 3. Phase 1 — Clean Training Engine

Đây là phần ưu tiên P0.

Protocol hiện đã quy định clean training:

* SGD
* LR 0.1
* momentum 0.9
* weight decay (5\times10^{-4})
* 200 epochs
* cosine scheduler
* RandomCrop
* RandomHorizontalFlip
* best checkpoint theo validation accuracy.

Do đó training implementation nên bám đúng specification này.

## 3.1 `src/aa/training/trainer.py`

Interface đề xuất:

```python
class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        checkpoint_manager,
    ):
        ...

    def train_epoch(self, loader, epoch):
        ...

    def evaluate(self, loader):
        ...

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        start_epoch=0,
    ):
        ...
```

Không gắn trainer với CIFAR cụ thể.

Nó chỉ biết:

```text
model
loader
optimizer
loss
scheduler
```

---

# 4. Train epoch phải trả metric rõ ràng

Một epoch nên produce:

```python
{
    "loss": ...,
    "accuracy": ...,
    "num_samples": ...,
    "learning_rate": ...,
    "runtime_seconds": ...
}
```

Validation:

```python
{
    "loss": ...,
    "accuracy": ...
}
```

History cuối:

```text
epoch
train_loss
train_accuracy
val_loss
val_accuracy
learning_rate
runtime
```

Không chỉ `print()` ra console.

---

# 5. `scripts/train_clean.py`

Script này nên rất mỏng.

CLI:

```bash
python scripts/train_clean.py \
    --config configs/train/clean_cifar10_resnet18.yaml
```

Optional:

```bash
--resume result/checkpoints/.../last.pth
--device cuda
--seed 42
```

Flow:

```text
read YAML
   ↓
validate config
   ↓
set seed
   ↓
create dataloaders
   ↓
create model
   ↓
optimizer
   ↓
scheduler
   ↓
Trainer.fit()
   ↓
save best + last
   ↓
evaluate best on test set
   ↓
save summary JSON
```

---

# 6. Clean-training config

Ví dụ:

```yaml
experiment:
  name: cifar10_resnet18_clean
  seed: 42

dataset:
  name: cifar10
  num_workers: 4

model:
  name: resnet18

training:
  epochs: 200
  train_batch_size: 256
  eval_batch_size: 512

optimizer:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0005
  nesterov: false

scheduler:
  name: cosine
  min_lr: 0.0

checkpoint:
  directory: result/checkpoints
  monitor: val_accuracy
  mode: max
  save_last: true
  save_best: true
```

Config này nên trở thành representation machine-readable của protocol.

---

# 7. Không nên reuse cùng batch size cho train/validation

Hiện `get_dataloaders()` nhận một `batch_size` chung.

Protocol lại muốn:

```text
train batch = 256
eval batch  = 512
```

Nên refactor thành:

```python
get_dataloaders(
    dataset_name,
    train_batch_size=256,
    eval_batch_size=512,
    ...
)
```

---

# 8. Phase 2 — Checkpoint system

Checkpoint không nên chỉ chứa:

```python
model.state_dict()
```

Mỗi checkpoint nên chứa:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,

    "epoch": ...,
    "best_val_accuracy": ...,

    "dataset": "cifar10",
    "architecture": "resnet18",

    "training_seed": 42,
    "training_config": {...},

    "git_commit": "...",
}
```

Sau khi file được save, tính:

```text
SHA256
```

Project hiện đã có helper để hash checkpoint và `get_model()` cũng lưu `checkpoint_sha256`.

Nên tận dụng.

---

# 9. Best và Last checkpoint phải tách

Ví dụ:

```text
result/checkpoints/
└── cifar10_resnet18_seed42/
    ├── best.pth
    ├── last.pth
    ├── config.yaml
    ├── history.json
    └── summary.json
```

### `best.pth`

Model có validation accuracy cao nhất.

### `last.pth`

Dùng resume training.

Không nên resume từ `best.pth` mặc định.

---

# 10. Resume training

P0 cho reproducibility khi train 200 epochs.

Resume phải restore:

```text
model
optimizer
scheduler
epoch
best metric
RNG states
```

Lý tưởng lưu:

```python
torch.get_rng_state()
torch.cuda.get_rng_state_all()
numpy RNG state
python random state
```

Như vậy resume gần deterministic nhất có thể.

---

# 11. Phase 3 — Checkpoint validation

Thêm:

```text
scripts/evaluate_checkpoint.py
```

Ví dụ:

```bash
python scripts/evaluate_checkpoint.py \
    --checkpoint result/checkpoints/.../best.pth \
    --dataset cifar10
```

Output:

```json
{
  "architecture": "resnet18",
  "dataset": "cifar10",
  "checkpoint_sha256": "...",
  "validation_accuracy": 94.7,
  "test_accuracy": 94.5
}
```

Mục đích là checkpoint phải được **certify trước khi attack benchmark sử dụng**.

---

# 12. Benchmark không nên nhận checkpoint không validated

Có thể thêm metadata:

```python
model.validation_accuracy
model.test_accuracy
```

Benchmark result sau này lưu:

```json
"model": {
    "architecture": "resnet18",
    "dataset": "cifar10",
    "clean_test_accuracy": 94.5,
    "sha256": "..."
}
```

Điều này giúp paper trace được chính xác model nào sinh ra bảng nào.

---

# 13. Phase 4 — Adversarial Training abstraction

Sau clean training mới implement AT.

Không nên viết ba script:

```text
train_pgd_at.py
train_spgd_at.py
train_sfa_at.py
```

Nên có **một**:

```text
train_adversarial.py
```

và abstraction:

```python
class AdversarialExampleGenerator:
    def generate(self, model, x, y):
        ...
```

Training flow:

```text
clean batch x
     ↓
attack/generator
     ↓
x_adv
     ↓
train objective
```

---

# 14. Ba AT method nên hỗ trợ

## AT-1 — PGD-(L_\infty)

Đây là defense reference chuẩn.

Threat model:

[
|\delta|_\infty\le 8/255.
]

Training attack có thể:

```text
PGD-10
eps = 8/255
alpha = 2/255
random start
```

Không cần dùng PGD-20/50 trong training vì quá đắt.

---

# 15. AT-2 — Sparse-PGD / PGD0 adversarial training

Đây mới thực sự liên quan trực tiếp tới Sparse AA.

Ví dụ:

[
L_0\le16.
]

Có thể thử:

```text
K = 8
K = 16
K sampled dynamically
```

Phương án thú vị hơn:

[
K \sim {4,8,16}
]

mỗi batch.

Như vậy model không overfit một support budget duy nhất.

---

# 16. AT-3 — SFA adversarial training

Đây là experiment rất quan trọng nhưng chỉ làm **sau khi proposed SFA đã freeze**.

Không dùng full paper SFA trong mỗi training batch vì:

* feature guidance;
* interaction;
* pruning;
* 25 steps;
* support deletion;

sẽ cực đắt.

Nên tạo:

```text
SFA-Train
```

là fast variant:

```yaml
steps: 5
pruning: false
feature_guidance: true
interaction: true
```

Trong paper phải ghi rõ:

> training attack là computationally reduced variant.

Không giả vờ rằng full evaluation SFA được sử dụng khi thực tế không phải.

---

# 17. Objective adversarial training

Support ít nhất hai mode.

## Pure adversarial training

[
L =
CE(f(x_{adv}),y).
]

## Mixed training

[
L =
(1-\lambda)
CE(f(x),y)
+
\lambda
CE(f(x_{adv}),y).
]

Default có thể:

[
\lambda=0.5.
]

Mixed objective đáng thử với sparse attacks vì sparse perturbations thường có distribution rất khác dense (L_\infty).

---

# 18. Sau đó có thể mở rộng TRADES

P1, không cần implementation đầu tiên.

[
L =
CE(f(x),y)
+
\beta
KL(f(x)|f(x_{adv})).
]

Có thể so:

```text
Standard AT
TRADES
Sparse AT
```

Nhưng không nên làm ở Phase đầu vì sẽ mở scope quá rộng.

---

# 19. Config adversarial training

Ví dụ PGD:

```yaml
experiment:
  name: cifar10_resnet18_pgd_at
  seed: 42

dataset:
  name: cifar10

model:
  name: resnet18

training:
  epochs: 200
  train_batch_size: 128

optimizer:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0005

adversarial_training:
  enabled: true
  attack: pgd

  clean_weight: 0.5
  adversarial_weight: 0.5

  attack_kwargs:
    eps: 0.031372549
    alpha: 0.007843137
    steps: 10
    random_start: true
```

Sparse:

```yaml
adversarial_training:
  attack: spgd

  attack_kwargs:
    k: 16
    steps: 10
```

---

# 20. Không dùng paper attack registry nguyên xi cho training

Registry hiện tại rất phù hợp benchmark.

Nhưng training có requirements khác:

```text
speed
memory
no unnecessary metric counting
no pruning
no CPU conversion
batch friendly
```

Ví dụ official PGD0 adapter convert tensor sang NumPy CPU. Không nên dùng nó trong mỗi training batch.

Vì vậy nên phân biệt:

```text
evaluation attack
training attack
```

Dù cùng family.

---

# 21. Phase 5 — Memory / AMP support

WRN-28-10 khá nặng.

Training system nên hỗ trợ:

```yaml
training:
  amp: true
```

sử dụng:

```python
torch.autocast
GradScaler
```

Nhưng adversarial-example generation cần cẩn thận.

Khuyến nghị:

```text
Attack gradient: FP32
Model update: AMP optional
```

để tránh gradient attack bị yếu do low precision.

---

# 22. Gradient accumulation

Nên có:

```yaml
gradient_accumulation_steps: 1
```

để sau này train WRN trên GPU memory thấp hơn.

Effective batch:

[
B_{effective}
=============

B_{physical}
\times
N_{accum}.
]

---

# 23. Phase 6 — Training reproducibility

Hiện project đã có `set_seed()` và deterministic cuDNN setup.

Training cần bổ sung worker seeding.

Ví dụ:

```python
worker_init_fn
generator=torch.Generator().manual_seed(seed)
```

DataLoader shuffle cũng cần deterministic seed.

### Metadata bắt buộc

Mỗi training result lưu:

```text
seed
git commit
git dirty
Python version
PyTorch version
CUDA version
cuDNN version
device
dataset
dataset split hash
checkpoint hash
config hash
```

---

# 24. Phase 7 — Logging

Không cần bắt buộc WandB ngay từ đầu.

P0 nên support:

```text
console
JSON
CSV
```

Ví dụ:

```text
history.csv
```

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
| ----: | ---------: | --------: | -------: | ------: | -: |

Sau này optional:

```yaml
logging:
  wandb: false
  tensorboard: true
```

---

# 25. Early stopping

Protocol yêu cầu 200 epochs, nên main paper training **không nên early-stop mặc định**.

Nhưng có thể:

```yaml
early_stopping:
  enabled: false
```

cho development.

Best checkpoint vẫn lưu bình thường.

---

# 26. Phase 8 — Test suite

Training code bắt buộc có tests trước khi chạy 200 epoch thật.

## `test_training.py`

Dùng tiny synthetic dataset:

```text
32 samples
2 epochs
```

Verify:

```text
training executes
weights change
loss finite
history length correct
checkpoint created
```

---

# 27. Checkpoint test

`test_checkpoint.py`:

1. train 1 epoch;
2. save;
3. reload;
4. verify logits giống nhau;
5. resume;
6. verify epoch counter đúng.

Ví dụ invariant:

[
f_{\theta_{saved}}(x)
=====================

f_{\theta_{loaded}}(x).
]

---

# 28. Determinism test

Cùng:

```text
config
seed
dataset
```

chạy tiny training hai lần.

Verify weights hoặc metrics giống nhau trong tolerance.

Đây là test rất có giá trị cho research artifact.

---

# 29. Adversarial training test

`test_adv_training.py` phải verify:

```text
x_adv.shape == x.shape
x_adv ∈ [0,1]
attack respects threat model
gradient flows into model update
attack graph detached before optimizer step as intended
```

Đối với (L_0):

[
L_0(x_{adv}-x)\le K.
]

---

# 30. Không được update model trong quá trình generate adversarial example

Đây là lỗi implementation khá dễ mắc.

Trong:

```text
generate x_adv
```

cần gradient:

[
\nabla_x L
]

nhưng không muốn accumulate:

[
\nabla_\theta L
]

vào optimizer state.

Phải quản lý:

```python
model.zero_grad()
x.requires_grad_(True)
```

và detach adversarial example trước training forward cuối:

```python
x_adv = x_adv.detach()
```

---

# 31. Phase 9 — Training sanity checks

Trước khi train full 200 epochs:

### Sanity A

Overfit 32 training examples.

Target gần:

```text
~100% train accuracy
```

Nếu không đạt → training pipeline có bug.

### Sanity B

Train CIFAR-10 5–10 epochs.

Loss phải giảm.

### Sanity C

Full ResNet-18 clean model.

Clean accuracy phải đạt vùng hợp lý trước khi attack benchmark.

Không chạy sparse attack benchmark nếu clean classifier quá yếu.

---

# 32. Checkpoint quality gate

Protocol đã support ý tưởng `min_clean_acc`.

Nên operationalize nó:

```yaml
quality_gate:
  min_validation_accuracy: 90.0
  min_test_accuracy: 90.0
```

Ví dụ model fail quality gate thì:

```text
TRAINING FAILED QUALITY GATE
```

và không được promote thành paper checkpoint.

Threshold chính xác sẽ xác định sau khi baseline ổn định.

---

# 33. Phase 10 — Clean model experiment matrix

Không train mọi thứ ngay.

## Stage A — Primary

```text
CIFAR-10
└── ResNet-18
```

Đây phải hoàn thiện đầu tiên.

## Stage B — Architecture generalization

```text
CIFAR-10
├── ResNet-18
└── WRN-28-10
```

## Stage C — Dataset generalization

```text
CIFAR-100
├── ResNet-18
└── WRN-28-10
```

ResNet-50 để supplementary sau.

Protocol cũng đang chọn ResNet-18 primary và WRN-28-10 secondary.

---

# 34. Adversarial-training experiment matrix

Sau clean models:

| Model    | Training      | Evaluation         |
| -------- | ------------- | ------------------ |
| ResNet18 | Clean         | all attacks        |
| ResNet18 | PGD-AT        | all attacks        |
| ResNet18 | Sparse-PGD-AT | all sparse attacks |
| ResNet18 | SFA-AT        | all sparse attacks |

Sau đó mới mở rộng WRN.

Không nên train 4 models × 2 datasets × 5 seeds ngay từ đầu.

---

# 35. Cross-attack evaluation

Đây là experiment defense quan trọng nhất.

Train:

[
T\in
{
Clean,
PGD,
SparsePGD,
SFA
}.
]

Attack:

[
A\in
{
PGD0,
SparsePGD,
SparseRS,
SigmaZero,
GSE,
SFA
}.
]

Ta có matrix:

[
R(T,A).
]

Ví dụ:

| Training      | PGD0 | SPGD | Sparse-RS | Sigma-0 | SFA |
| ------------- | ---: | ---: | --------: | ------: | --: |
| Clean         |      |      |           |         |     |
| PGD-AT        |      |      |           |         |     |
| Sparse-PGD-AT |      |      |           |         |     |
| SFA-AT        |      |      |           |         |     |

Matrix này trả lời được defense generalization thay vì chỉ:

> “SFA-AT chống được SFA”.

---

# 36. Defense evaluation phải tách hai loại

Sau khi thêm AT:

## Preprocessing defenses

Hiện đã có:

```text
Blur
Median
JPEG
TV
```

## Learned defenses

```text
PGD-AT
Sparse-PGD-AT
SFA-AT
```

Paper nên tách hai nhóm.

Không xem preprocessing và AT là cùng một loại defense.

---

# 37. Config hiện tại cần refactor

`configs/paper.yaml` hiện trộn:

```text
dataset
model
benchmark
attacks
defense attacks
```

Sau training subsystem, không nên tiếp tục nhét tất cả vào một config.

Tách:

```text
train config
benchmark config
defense config
```

Ví dụ:

```text
configs/train/
configs/benchmark/
configs/defense/
```

Điều này giảm coupling rất nhiều.

---

# 38. P0 — strict config validation

Nên thêm dataclass/Pydantic hoặc manual schema validation.

Ví dụ typo:

```yaml
learning_ratre: 0.1
```

phải fail.

Không được quietly dùng default.

Tương tự attack config.

Research configs phải:

> **fail loudly, never silently fallback.**

---

# 39. Đề xuất training module chi tiết

```text
src/aa/training/
│
├── trainer.py
│   ├── Trainer
│   ├── train_epoch()
│   └── validation_epoch()
│
├── adversarial.py
│   ├── AdversarialTrainer
│   ├── PGDTrainAttack
│   ├── SparsePGDTrainAttack
│   └── SFATrainAttack
│
├── optim.py
│   ├── create_optimizer()
│   └── create_scheduler()
│
├── checkpoint.py
│   ├── save_checkpoint()
│   ├── load_checkpoint()
│   ├── resume_checkpoint()
│   └── CheckpointManager
│
└── history.py
    ├── TrainingHistory
    └── save_history()
```

---

# 40. `models.py` nên chỉ làm model factory

Không nên thêm training logic vào `models.py`.

Giữ:

```text
architecture definition
model factory
checkpoint model loading for evaluation
```

Có thể tách về lâu dài:

```text
models/
    resnet.py
    wideresnet.py
    registry.py
```

nhưng chưa phải P0.

---

# 41. `data.py` cần thay đổi gì

Hiện data pipeline đã khá gần yêu cầu.

Cần sửa:

1. train/eval batch size riêng;
2. dataset metadata;
3. remove unnecessary CIFAR eval resize;
4. deterministic DataLoader worker seed;
5. expose train/val split hash;
6. optional `num_workers`;
7. optional `pin_memory`;
8. optional persistent workers.

Không cần rewrite toàn bộ.

---

# 42. Output training chuẩn

Sau một training run:

```text
result/training/
└── cifar10_resnet18_clean_seed42/
    ├── config.yaml
    ├── best.pth
    ├── last.pth
    ├── history.csv
    ├── history.json
    ├── summary.json
    └── environment.json
```

`summary.json`:

```json
{
  "dataset": "cifar10",
  "architecture": "resnet18",

  "seed": 42,

  "best_epoch": 174,
  "best_validation_accuracy": 94.8,
  "test_accuracy": 94.5,

  "checkpoint_sha256": "...",
  "git_commit": "..."
}
```

---

# 43. Paper checkpoint promotion

Nên có khái niệm:

```text
candidate checkpoint
        ↓
validation
        ↓
paper checkpoint
```

Chỉ checkpoint nào đạt:

```text
config complete
test success
hash recorded
clean accuracy threshold
git clean/reproducible
```

mới được benchmark chính.

Có thể đặt:

```text
result/paper_checkpoints/
```

hoặc metadata:

```json
"paper_eligible": true
```

---

# 44. Không chọn checkpoint bằng test accuracy

Protocol hiện đúng khi dùng validation cho model selection.

Flow phải luôn:

```text
Train
  ↓
Validation → choose best
  ↓
Test → report once
```

Tuyệt đối không:

```text
Test every epoch
→ choose best test accuracy
```

vì test leakage.

---

# 45. Seeds

Training primary đầu tiên:

```text
seed = 42
```

để debug.

Paper final nên ít nhất:

```text
3 independent training seeds
```

ví dụ:

[
{42,43,44}.
]

Nếu compute đủ:

[
5 seeds.
]

Attack stochastic seeds lại là một dimension khác.

Phải phân biệt:

```text
training seed
attack seed
sampling seed
```

---

# 46. Naming checkpoint

Không dùng generic:

```text
best.pth
```

ở root.

Artifact name nên encode:

```text
dataset
architecture
training method
seed
```

Ví dụ:

```text
resnet18_cifar10_clean_seed42_best.pth
resnet18_cifar10_pgd_at_seed42_best.pth
resnet18_cifar10_spgd_at_k16_seed42_best.pth
```

---

# 47. README cần bổ sung

Quick start cuối cùng:

```bash
# Install
pip install -e .

# Train clean model
python scripts/train_clean.py \
  --config configs/train/clean_cifar10_resnet18.yaml

# Train adversarial model
python scripts/train_adversarial.py \
  --config configs/adv_train/pgd_at_cifar10_resnet18.yaml

# Benchmark
python scripts/attack_benchmark.py \
  --config configs/benchmark/cifar10_resnet18.yaml
```

Hiện README chỉ hướng dẫn attack/defense benchmark và pytest.

---

# 48. Protocol cần bổ sung adversarial training section

Hiện protocol đã nêu clean training khá rõ.

Nên thêm:

```text
Adversarial Training Protocol
├── training threat model
├── attack steps
├── epsilon / K
├── clean/adv loss mix
├── evaluation attacks
├── model selection rule
└── compute budget
```

Đặc biệt phải tách:

```text
training attack strength
evaluation attack strength
```

Evaluation luôn mạnh hơn training attack.

---

# 49. Thứ tự triển khai mình đề xuất

## P0.1 — Dataset/model compatibility

```text
dataset specs
num_classes automatic
batch sizes
```

## P0.2 — Clean trainer core

```text
Trainer
optimizer
scheduler
metrics
```

## P0.3 — Checkpoint/resume

```text
best
last
metadata
SHA256
```

## P0.4 — Clean CLI

```text
train_clean.py
```

## P0.5 — Tests

```text
training
resume
determinism
```

## P0.6 — Train CIFAR10 ResNet18

Đây là milestone đầu tiên.

---

# 50. Sau milestone Clean ResNet18

```text
P1.1 WRN28-10 clean
      ↓
P1.2 PGD adversarial training
      ↓
P1.3 Sparse-PGD adversarial training
      ↓
P1.4 Defense cross-attack benchmark
      ↓
P1.5 Freeze SFA
      ↓
P1.6 SFA adversarial training
      ↓
P1.7 CIFAR-100
```

Mình đặc biệt khuyên **không implement SFA-AT ngay bây giờ**.

Vì proposed attack hiện vẫn có thể thay đổi. Nếu SFA thay đổi sau này, SFA-trained checkpoints cũ sẽ trở thành artifacts khó diễn giải.

---

# 51. Dependency graph

```text
Dataset refactor
        │
        ▼
Clean Trainer
        │
        ├──────────► Checkpoint Manager
        │
        └──────────► Logging
        │
        ▼
ResNet18 Clean Checkpoint
        │
        ├──────────► Attack Benchmark
        │
        ▼
Adversarial Trainer
        │
    ┌───┴────┐
    ▼        ▼
 PGD-AT   Sparse-AT
    │        │
    └───┬────┘
        ▼
Cross-Attack Defense Evaluation
        │
        ▼
Freeze Proposed Method
        │
        ▼
SFA-AT
```

---

# 52. Definition of Done — clean training

Clean subsystem chỉ được coi là hoàn thành khi:

* [ ] ResNet-18/CIFAR-10 train từ scratch được.
* [ ] Validation split deterministic.
* [ ] Best checkpoint được chọn bằng validation accuracy.
* [ ] Resume checkpoint chạy đúng.
* [ ] Checkpoint lưu full metadata.
* [ ] SHA256 được record.
* [ ] Test accuracy được tính sau model selection.
* [ ] Cùng seed cho reproducible result.
* [ ] Unit tests pass.
* [ ] Benchmark load trực tiếp checkpoint vừa train được.
* [ ] Không cần manual edit code để chuyển ResNet18 → WRN.
* [ ] Không cần manual edit code để chuyển CIFAR10 → CIFAR100.

---

# 53. Definition of Done — adversarial training

* [ ] PGD-AT training chạy ổn.
* [ ] Sparse-AT training chạy ổn.
* [ ] Training attack tuân thủ threat model.
* [ ] Clean accuracy được report.
* [ ] Robust accuracy được report.
* [ ] Không gradient leakage từ attack generation.
* [ ] Robust checkpoint load được bởi benchmark hiện tại.
* [ ] Adaptive attack evaluation được chạy.
* [ ] Cross-attack matrix được tạo.
* [ ] Training configuration + seed + hashes được lưu.
* [ ] SFA-AT chỉ triển khai sau khi proposed method freeze.

---

# 54. Deliverable cuối cùng

Khi hoàn thiện plan này, AA_2 sẽ không còn là:

```text
pretrained checkpoint
       ↓
benchmark attacks
```

mà thành một research pipeline hoàn chỉnh:

[
\boxed{
Data
\rightarrow
Training
\rightarrow
Checkpoint\ Verification
\rightarrow
Attack
\rightarrow
Defense
\rightarrow
Robustness\ Evaluation
}
]

Và quan trọng hơn, mọi số trong paper có thể trace ngược:

[
Result
\rightarrow
Checkpoint
\rightarrow
Training\ Config
\rightarrow
Seed
\rightarrow
Dataset\ Split
\rightarrow
Git\ Commit.
]

### Mức ưu tiên triển khai thực tế

**Sprint 1:** dataset/model refactor → clean trainer → checkpoint/resume → tests → CIFAR-10 ResNet-18.

**Sprint 2:** WRN-28-10 → PGD-AT → Sparse-PGD-AT → cross-attack defense.

**Sprint 3:** freeze SFA → SFA-AT → CIFAR-100 → multi-seed experiments → final paper tables.

Đây là thứ tự mình khuyến nghị để tránh việc dành compute cho adversarial training trước khi **clean training và proposed attack đã ổn định**.
