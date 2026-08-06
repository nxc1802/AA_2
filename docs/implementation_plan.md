# Sparse Adversarial Attack Full Pipeline on Marimo GPU Server

Pipeline hoàn chỉnh: Kết nối Marimo → Tải CIFAR-10 từ HF → Train ResNet-18 → Attack Benchmark (Baseline + 19 SOTA + 4 Proposed) với Ablation K → Xuất CSV/JSON + Hẹn giờ 15 phút.

## User Review Required

> [!IMPORTANT]
> - **Execution Target**: Remote Marimo server tại docs/marimo.txt (GPU server).
> - **Pipeline gồm 4 Phase lớn**, mỗi Phase là một script Python riêng biệt được gửi lên server qua `execute-code.sh`.

> [!WARNING]
> - HF Token được đọc từ môi trường hoặc HF Hub login.

---

## Proposed Changes

### Phase 0: Kết Nối & Toast Notification

#### [NEW] `scratch/phase0_connect_toast.py`
- Kết nối session mới tới Marimo server.
- Gửi toast notification `mo.status.toast(...)` thông báo sẵn sàng pair-program.
- Kiểm tra GPU availability (`torch.cuda.is_available()`, `torch.cuda.get_device_name()`).
- Kiểm tra installed packages (torch, torchvision, datasets, sklearn, scipy, pandas).

---

### Phase 1: Tải Dữ Liệu CIFAR-10 từ Hugging Face

#### [NEW] `scratch/phase1_load_data.py`
- Tải full CIFAR-10 (50,000 train + 10,000 test) từ HF repo `Cuong2004/AA`.
- Thực hiện stratified split 40,000 train / 10,000 val với `seed=42`.
- Verify data shapes và class distribution.
- Lưu split indices cho reproducibility.

---

### Phase 2: Train Clean ResNet-18 Checkpoint

#### [NEW] `scratch/phase2_train_resnet18.py`
- CIFAR-adapted ResNet-18: conv1 = 3×3 stride=1, bỏ maxpool, fc → 10 classes.
- Training recipe (theo [requirement.md](file:///Volumes/WorkSpace/Project/Bản sao AA/docs/requirement.md)):
  - SGD, momentum=0.9, weight_decay=5e-4
  - batch_size=128, 200 epochs
  - lr=0.1, CosineAnnealingLR
  - RandomCrop(32, padding=4) + RandomHorizontalFlip
- Lưu best checkpoint (highest val accuracy).
- Report: Clean Accuracy trên 10,000 test images.

---

### Phase 3: Attack Phase & Ablation K

#### [NEW] `scratch/phase3_attack_ablation.py`

Đây là Phase lớn nhất, implement inline tất cả attacks và chạy ablation.

**3.1 Baseline Attacks (Dense)**:
- FGSM (ε=8/255, 1 step)
- BIM (ε=8/255, 10 steps, α=2/255)
- PGD (ε=8/255, 20 steps, α=2/255, random start)

> Dense attacks chạy 1 lần duy nhất (không vary K), kết quả dùng làm baseline comparison.

**3.2 SOTA Sparse Attacks (19 methods) × K ∈ {1, 2, 4, 8, 16, 32, 64, 128}**:

| Category | Attack | K-parameterizable? |
|:---|:---|:---|
| Classical | JSMA | ✅ max_pixels=K |
| Classical | One Pixel Attack | ✅ pixels=K |
| Classical | SparseFool | ✅ max_perturbed=K |
| Classical | CornerSearch | ✅ max_pixels=K |
| Optimization | SAIF | ✅ k=K |
| Optimization | σ-zero | ✅ sparsity=K |
| Optimization | Homotopy | ✅ target_sparsity=K |
| Optimization | PGD0 | ✅ k=K |
| Optimization | GSE | ✅ group_budget=K |
| Optimization | Sparse-PGD | ✅ sparsity_budget=K |
| Attention | IPFSA | ✅ k_pixels=K |
| Attention | Gradient Guidance | ✅ sparsity_budget=K |
| Frequency | SFA | ✅ freq_k=K |
| Black-box | Sparse-RS | ✅ n_pixels=K |
| Black-box | BruSLeAttack | ✅ budget=K |
| Black-box | Pixle | ✅ n_swaps=K |

> Mỗi attack × mỗi K: chạy trên toàn bộ 10,000 test images.

**3.3 Proposed Methods (4 methods) × K ∈ {1, 2, 4, 8, 16, 32, 64, 128}**:
- CPA: `coalition_size=K`
- FCSA: `max_coalition_size=K`
- FMSA: `support_budget=K`
- HSA: `budget=K`

**3.4 Metrics tính cho mỗi (Attack, K)**:

| Metric Group | Metrics |
|:---|:---|
| Effectiveness | Clean Accuracy, Robust Accuracy, ASR (conditional), Accuracy Drop, Fooling Rate |
| Perturbation | L₀, L₀/(H×W), L₂, L∞, Mean perturbation per changed pixel |
| Image Quality | PSNR, SSIM, LPIPS (nếu có `lpips` package) |
| Efficiency | Runtime/image, Forward/Backward passes |

---

### Phase 4: Lưu File & Báo Cáo

#### [NEW] `scratch/phase4_save_report.py`
- **ASR–K Curve Pivot Table**: Rows = Attack Methods, Columns = K values, Values = ASR(%).
- **Robust Accuracy–K Curve Pivot Table**: tương tự.
- **Full metrics DataFrame**: tất cả metrics per (Attack, K).
- Lưu vào `/marimo/result/metrics/`:
  - `asr_k_pivot.csv` / `asr_k_pivot.json`
  - `robust_accuracy_k_pivot.csv` / `robust_accuracy_k_pivot.json`
  - `full_attack_metrics.csv` / `full_attack_metrics.json`
  - `image_quality_metrics.csv` / `image_quality_metrics.json`
- Toast notification khi hoàn tất.
- **Hẹn giờ 15 phút**: Sử dụng `schedule` tool với `CronExpression="*/15 * * * *"` để kiểm tra tiến độ định kỳ.

---

## Execution Strategy

1. **Chạy tuần tự**: Phase 0 → 1 → 2 → 3 → 4, mỗi Phase gửi qua `execute-code.sh`.
2. **Script Phase 3 sẽ rất lớn** (~2000-3000 lines) vì chứa cả implementation lẫn evaluation. Sẽ gửi qua file (không dùng `-c`).

## Verification Plan

### Automated Tests
- Phase 0: Verify GPU detection + toast delivered.
- Phase 1: Assert `len(train)=40000`, `len(val)=10000`, `len(test)=10000`.
- Phase 2: Assert `val_accuracy > 90%`, `test_accuracy > 90%` (expected ~93-95% for CIFAR-10 ResNet-18).
- Phase 3: Verify ASR values are in range [0, 100], L₀ ≤ K, tổng results = (3 baselines + 19×8 + 4×8) = 3 + 152 + 32 = 187 rows.
- Phase 4: Verify CSV/JSON files exist and are valid.

### Manual Verification
- Kiểm tra toast notifications trên Marimo UI.
- Xem ASR–K Curve Pivot Table để đánh giá proposed methods vs SOTA.
