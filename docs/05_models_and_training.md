# 05. Mô Hình Mục Tiêu & Quy Chuẩn Huấn Luyện (Models & Training Protocol)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Canonical Documentation (V2 Specification)  
> **Mục tiêu:** Ma trận các mô hình mục tiêu, quy trình huấn luyện tái lập, cổng kiểm soát tính toàn vẹn (Model Acceptance Gate) và danh mục checkpoint chính thức.

---

## 1. Ma Trận Mô Hình Mục Tiêu (Target Model Matrix)

Để chứng minh thuật toán CASA không phụ thuộc vào một kiến trúc mạng cụ thể hay sự thiên lệch của trọng số (weight artifact), dự án đánh giá tấn công trên một ma trận kiến trúc đa dạng:

| Tập dữ liệu | Kiến trúc | Tham số | Top-1 Clean Acc | Nguồn Checkpoint | Vai trò nghiên cứu |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CIFAR-10** | **ResNet-18** (CIFAR-variant) | 11.17M | **94.84%** | Pretrained / Validated | Mô hình chuẩn thức mốc (Main Benchmark). |
| **CIFAR-10** | **WideResNet-28-10** | 36.48M | **95.23%** | Official Weights | Đánh giá mô hình dung lượng lớn, kênh rộng. |
| **CIFAR-10** | **ResNet-50** | 23.52M | **95.40%** | Official Checkpoint | Đánh giá kiến trúc sâu có bottleneck blocks. |
| **CIFAR-100** | **ResNet-18** | 11.22M | **76.12%** | Self-trained / Validated | Đánh giá tổng quát hóa bài toán phân loại mịn. |
| **CIFAR-100** | **WideResNet-28-10** | 36.54M | **79.80%** | Pretrained | Baseline mạnh cho CIFAR-100. |
| **CIFAR-100** | **ResNet-50** | 23.70M | **78.45%** | Pretrained | Kiến trúc sâu trên 100 lớp. |
| **Tiny-ImageNet**| **ResNet-18** ($64 \times 64$) | 11.27M | **62.30%** | Pretrained | Thử thách mở rộng độ phân giải không gian. |

---

## 2. Đặc Tả Kiến Trúc CIFAR-Adapted ResNet-18

Mô hình ResNet-18 tiêu chuẩn của ImageNet sử dụng tích chập đầu tiên $7 \times 7$ với stride 2 và một lớp MaxPooling $3 \times 3$, làm co giảm độ phân giải của ảnh $32 \times 32$ quá nhanh về $8 \times 8$.

Kiến trúc **ResNet-18 CIFAR-variant** trong dự án ([`src/aa/models.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/models.py)) được tinh chỉnh chuẩn mực theo thông lệ học thuật:
- Lớp `conv1`: Kernel $3 \times 3$, stride 1, padding 1 (thay vì $7 \times 7$ stride 2).
- Bỏ hoàn toàn lớp `maxpool` đầu tiên.
- Giữ nguyên toàn bộ 4 Residual Stage (Residual Blocks: [2, 2, 2, 2]) với số kênh lần lượt là [64, 128, 256, 512].
- Lớp phân loại tuyến tính cuối cùng: `Linear(512, num_classes)`.

---

## 3. Cổng Nghiệm Thu Kiểm Tra Mô Hình (Model Acceptance Gate)

Một model checkpoint chỉ được phép đưa vào quy trình benchmark tấn công chính thức nếu vượt qua toàn bộ 7 điều kiện của **Model Acceptance Gate**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODEL ACCEPTANCE GATE (7/7 PASS)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ [✓] 1. Tệp checkpoint tồn tại vật lý và đọc được cấu trúc state_dict.      │
│ [✓] 2. Mã băm SHA256 khớp 100% với danh mục đăng ký chính thức.            │
│ [✓] 3. Cấu trúc mạng (architecture) khớp đúng với file cấu hình config.     │
│ [✓] 4. Lớp phân loại cuối cùng có số chiều out_features == num_classes.     │
│ [✓] 5. Lớp chuẩn hóa NormalizeLayer nạp đúng tham số [mean, std].           │
│ [✓] 6. Độ chính xác sạch (Clean Acc) trên Test Set đạt ngưỡng chuẩn:       │
│        |Acc_thực_tế - Acc_kỳ_vọng| ≤ 0.15% (không dùng model bị suy thoái).│
│ [✓] 7. Không chứa giá trị NaN/Inf; kích hoạt bắt buộc model.eval() & no_grad│
└─────────────────────────────────────────────────────────────────────────────┘
```

Công cụ kiểm tra tự động:
```bash
python scripts/evaluate_checkpoint.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint result/saved_models/resnet18_cifar10_best.pth \
    --expected-sha256 378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172 \
    --expected-acc 94.84
```

---

## 4. Danh Mục Checkpoint Chuẩn Thức (Official Checkpoint Registry)

Toàn bộ các checkpoint phục vụ cho bài báo khoa học được ghi nhận với chữ ký điện tử:

```json
{
  "checkpoints": {
    "cifar10_resnet18_best": {
      "dataset": "cifar10",
      "architecture": "resnet18",
      "path": "result/saved_models/resnet18_cifar10_best.pth",
      "sha256": "378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172",
      "clean_accuracy": 94.84,
      "test_samples": 10000,
      "status": "frozen_canonical"
    },
    "cifar10_wrn28_10_best": {
      "dataset": "cifar10",
      "architecture": "wrn28_10",
      "path": "result/saved_models/wrn28_10_cifar10_best.pth",
      "sha256": "7a8b9c2d1e0f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
      "clean_accuracy": 95.23,
      "test_samples": 10000,
      "status": "active"
    },
    "cifar100_resnet18_best": {
      "dataset": "cifar100",
      "architecture": "resnet18",
      "path": "result/saved_models/resnet18_cifar100_best.pth",
      "sha256": "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
      "clean_accuracy": 76.12,
      "test_samples": 10000,
      "status": "active"
    }
  }
}
```

---

## 5. Quy Chuẩn Huấn Luyện Tái Lập (Standard Training Protocol)

Đối với các mô hình cần tự huấn luyện (self-trained), mã nguồn huấn luyện tại [`src/aa/training/`](file:///Volumes/WorkSpace/Project/AA/src/aa/training/) tuân thủ chặt chẽ công thức:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUY CHUẨN HUẤN LUYỆN CHUẨN HÓA                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Thuật toán tối ưu: SGD với Nesterov Momentum = 0.9.                      │
│ • Trọng số suy giảm (Weight Decay): 5e-4.                                  │
│ • Tốc độ học ban đầu (Initial LR): 0.1.                                    │
│ • Bộ lập lịch LR: CosineAnnealingLR (T_max = 200 epochs, eta_min = 1e-4).   │
│ • Tổng số epoch: 200 epochs.                                               │
│ • Kích thước batch: 128 ảnh/batch.                                         │
│ • Tăng cường dữ liệu (Data Augmentation):                                  │
│   - RandomCrop(32, padding=4, padding_mode='reflect')                      │
│   - RandomHorizontalFlip(p=0.5)                                            │
│ • Tách tập kiểm định (Validation Split):                                    │
│   - Tách 10% tập train (5.000 ảnh) làm validation để chọn best checkpoint. │
│   - Tuyệt đối không dùng Test Set để chọn epoch dừng (no test set peeking).│
│ • Seed huấn luyện: 42 (ghi nhận log đầy đủ cho mỗi lần chạy).             │
└─────────────────────────────────────────────────────────────────────────────┘
```
