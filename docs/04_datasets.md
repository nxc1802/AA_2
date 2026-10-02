# 04. Đặc Tả Dữ Liệu & Danh Mục Quản Lý (Datasets Specification & Registry)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Canonical Documentation (V2 Specification)  
> **Mục tiêu:** Quy định cấu hình tập dữ liệu, tiền xử lý, chuẩn hóa, lấy mẫu phân tầng và khóa tính toàn vẹn (cryptographic fingerprint) cho toàn bộ các tập kiểm chuẩn.

---

## 1. Ma Trận Tập Dữ Liệu Nghiên Cứu (Dataset Research Matrix)

Dự án mở rộng nghiên cứu từ tập cơ sở CIFAR-10 sang các bài toán có độ phức tạp cao hơn và không gian tìm kiếm lớn hơn:

| Tập dữ liệu | Số lớp | Độ phân giải | Kênh | Kích thước Test | Vai trò nghiên cứu | Trạng thái thực thi |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **CIFAR-10** | 10 | $32 \times 32$ | 3 | 10.000 | Phát triển chính, full benchmark 8 đối thủ, ablation 10 biến thể, multi-seed. | **Hoàn thành (10k/10k)** |
| **CIFAR-100** | 100 | $32 \times 32$ | 3 | 10.000 | Kiểm chứng tổng quát hóa khi số lớp tăng gấp 10 lần, biên quyết định phức tạp. | **Hoàn thành** |
| **Tiny-ImageNet** | 200 | $64 \times 64$ | 3 | 10.000 | Thử thách mở rộng quy mô không gian tìm kiếm ($HW: 1.024 \to 4.096$ pixel). | **Đang triển khai** |
| **ImageNet-100** | 100 | $224 \times 224$ | 3 | 5.000 | Stretch goal: Đánh giá khả năng mở rộng trên ảnh độ phân giải cao tiêu chuẩn. | **Dự kiến (Phase 4)** |

---

## 2. Đặc Tả Chi Tiết Từng Tập Dữ Liệu

### 2.1 CIFAR-10 (Canonical Development & Benchmark)
- **Nguồn:** `torchvision.datasets.CIFAR10`
- **Phiên bản:** PyTorch standard release
- **Tập kiểm thử (Test Split):** 10.000 ảnh tĩnh, kích thước $3 \times 32 \times 32$.
- **Mã băm danh sách mẫu (Sample Indices Hash):**
  $$\text{SHA256} = \texttt{1899ec16689e0477ac35484f5cacefc22e284a929c103400ef4f1638739aba08}$$
  Mã băm này khóa chặt thứ tự nạp của toàn bộ 10.000 ảnh mẫu, đảm bảo tái lập bit-to-bit giữa các môi trường (Local, Cluster, Kaggle GPU).
- **Phân phối nhãn:** Cân bằng hoàn hảo ($1.000$ ảnh/lớp trên 10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

### 2.2 CIFAR-100 (Fine-Grained Classification Challenge)
- **Nguồn:** `torchvision.datasets.CIFAR100`
- **Tập kiểm thử:** 10.000 ảnh, 100 lớp mịn (20 siêu lớp).
- **Vai trò:** Kiểm tra xem ưu thế của CASA ở vùng cực thưa ($K \le 2$) có phải là hệ quả phụ thuộc vào tính đơn giản của 10 lớp CIFAR-10 hay vẫn duy trì mạnh mẽ khi số lượng lớp cạnh tranh tăng vọt.

### 2.3 Tiny-ImageNet (Spatial Resolution Scalability)
- **Nguồn:** Stanford CS231N Tiny-ImageNet-200
- **Độ phân giải:** $64 \times 64$ pixel ($4.096$ điểm ảnh, gấp 4 lần CIFAR).
- **Tập kiểm thử:** 10.000 ảnh validation split chuẩn hóa.
- **Ý nghĩa khoa học:** Không gian tìm kiếm tổ hợp tăng lũy thừa theo kích thước ảnh:
  $$\binom{H \times W}{K} = \binom{4096}{K} \gg \binom{1024}{K}$$
  Đây là môi trường kiểm chứng trực tiếp tính hiệu quả của cơ chế sàng lọc Directional Headroom Gain và Spatial NMS.

---

## 3. Quy Chuẩn Tiền Xử Lý & Lớp Chuẩn Hóa (Normalization Protocol)

### 3.1 Không gian tối ưu hóa đối nghịch $[0, 1]$
Để tuân thủ tuyệt đối ràng buộc mô hình mối đe dọa, **mọi ảnh đầu vào cho các thuật toán tấn công đều nằm trong miền $[0, 1]$ thuần túy**:
$$x \in [0, 1]^{3 \times H \times W}$$
Tuyệt đối không áp dụng chuẩn hóa trừ trung bình và chia độ lệch chuẩn (Z-score normalization) trực tiếp trong Data Loader.

### 3.2 Lớp bao bọc chuẩn hóa nội tại (In-Model Normalization Wrapper)
Các tham số chuẩn hóa (mean và std) được đóng gói thành một lớp `NormalizeLayer` đặt ngay ở đầu kiến trúc mạng:

```python
import torch
import torch.nn as nn

class NormalizedModel(nn.Module):
    """Encapsulates input normalization within the model graph."""
    def __init__(self, backbone: nn.Module, mean: list, std: list):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be in [0, 1]
        x_norm = (x - self.mean) / self.std
        return self.backbone(x_norm)
```

**Bộ tham số chuẩn hóa chuẩn thức:**
- **CIFAR-10:**
  - $\mu = [0.4914, 0.4822, 0.4465]$
  - $\sigma = [0.2470, 0.2435, 0.2616]$
- **CIFAR-100:**
  - $\mu = [0.5071, 0.4867, 0.4408]$
  - $\sigma = [0.2675, 0.2565, 0.2761]$
- **Tiny-ImageNet:**
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$

Thiết kế này đảm bảo gradient $\nabla_x \mathcal{L}$ được tính toán tự động qua quy tắc chuỗi (chain rule) phản ánh đúng tỷ lệ co giãn của từng kênh màu mà không làm thay đổi miền giá trị hộp $[0, 1]$.

---

## 4. Chính Sách Lấy Mẫu Phân Tầng (Stratified Sampling Policy)

Để phục vụ các pha thực nghiệm với quy mô tài nguyên khác nhau, dự án quy định 3 mức kích thước mẫu dựa trên cơ chế phân tầng (Stratified Sampling) giữ nguyên tỷ lệ phân bố giữa các lớp nhãn:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CHÍNH SÁCH LẤY MẪU PHÂN TẦNG                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. SMOKE SET (N = 20 hoặc 32 mẫu):                                          │
│    • Mục đích: Kiểm thử sanity trong CI/CD, kiểm tra không lỗi runtime.    │
│    • Phân bổ: 2 hoặc 3 mẫu/lớp ngẫu nhiên có cố định seed.                 │
│                                                                             │
│ 2. DEVELOPMENT SET (N = 1.000 mẫu):                                         │
│    • Mục đích: Khám phá siêu tham số, phân rã thành phần (ablation).        │
│    • Phân bổ: 100 mẫu/lớp (CIFAR-10) hoặc 10 mẫu/lớp (CIFAR-100).          │
│                                                                             │
│ 3. CANONICAL TEST SET (N = 10.000 mẫu):                                     │
│    • Mục đích: Báo cáo kết quả bài báo chính thức (Official Paper Numbers). │
│    • Toàn bộ 10.000 mẫu của Test Split chuẩn hóa.                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

Mọi bộ chọn mẫu con đều được hiện thực tại [`src/aa/data.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/data.py) với hàm `get_cifar10_loaders` và cờ `--samples`.

---

## 5. Cấu Trúc Khai Báo Danh Mục Dữ Liệu (Dataset Registry Schema)

Tất cả các tập dữ liệu được khai báo tập trung dưới định dạng YAML/JSON chuẩn hóa:

```yaml
datasets:
  cifar10:
    name: "CIFAR-10"
    num_classes: 10
    image_size: [3, 32, 32]
    test_size: 10000
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2470, 0.2435, 0.2616]
    sample_indices_hash: "1899ec16689e0477ac35484f5cacefc22e284a929c103400ef4f1638739aba08"
    download: true

  cifar100:
    name: "CIFAR-100"
    num_classes: 100
    image_size: [3, 32, 32]
    test_size: 10000
    mean: [0.5071, 0.4867, 0.4408]
    std: [0.2675, 0.2565, 0.2761]
    download: true

  tiny_imagenet:
    name: "Tiny-ImageNet-200"
    num_classes: 200
    image_size: [3, 64, 64]
    test_size: 10000
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    download: false
```
