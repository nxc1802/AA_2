# 11. Sổ Tay Tái Lập Thực Nghiệm (Reproducibility Guide & Cookbook)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Authoritative Reproduction Cookbook (Consolidated Kaggle & Local Guide)  
> **Mục tiêu:** Cung cấp quy trình từng bước (step-by-step cookbook) để một nhà nghiên cứu độc lập có thể tái lập toàn bộ các bảng số liệu và hình vẽ trong bài báo từ một bản sao mã nguồn sạch.

---

## 1. Yêu Cầu Phần Cứng & Môi Trường (System Requirements)

- **Hệ điều hành:** Linux (Ubuntu 20.04/22.04 LTS khuyến nghị) hoặc macOS (Apple Silicon MPS hỗ trợ dev/smoke).
- **Python:** $\ge 3.10$ (kiểm chuẩn chính thức trên Python 3.10 và 3.11).
- **PyTorch:** $\ge 2.0.0$ với CUDA 11.8 hoặc 12.x.
- **Bộ nhớ RAM:** Tối thiểu 16 GB hệ thống.
- **GPU Khuyến nghị:**
  - *Chạy chính thức 10.000 mẫu:* 1× NVIDIA A100 (40GB/80GB) hoặc 2× NVIDIA Tesla T4 (Kaggle GPU miễn phí).
  - *Chạy thử nghiệm / phát triển:* 1× GPU $\ge 8$ GB VRAM (RTX 3070, T4, hoặc MPS).
- **Dung lượng ổ đĩa:** Tối thiểu 10 GB trống (cho checkpoint, dataset CIFAR và cache artifacts).

---

## 2. Quy Trình Tái Lập Từng Bước (10-Step Reproduction Cookbook)

### Bước 1: Sao chép kho lưu trữ & Khởi tạo môi trường ảo
```bash
git clone https://github.com/nxc1802/AA_2.git
cd AA_2

# Tạo và kích hoạt môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Cài đặt gói aa ở chế độ editable cùng các phụ thuộc
pip install --upgrade pip
pip install -e .
```

### Bước 2: Xác minh tính toàn vẹn của Model Checkpoint
Trước khi chạy bất kỳ thực nghiệm nào, bắt buộc phải kiểm tra mã băm SHA256 của checkpoint ResNet-18:
```bash
python scripts/evaluate_checkpoint.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint result/saved_models/resnet18_cifar10_best.pth \
    --expected-sha256 378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172 \
    --expected-acc 94.84
```
*Kết quả bắt buộc: `CHECKPOINT INTEGRITY VERIFIED (PASS)`.*

### Bước 3: Chạy Smoke Test xác nhận hệ thống (< 1 phút)
Chạy kiểm thử nhanh trên 20 mẫu ảnh để đảm bảo toàn bộ pipeline hoạt động trơn tru:
```bash
python scripts/attack_benchmark.py --config configs/smoke.yaml --strict
```

### Bước 4: Chạy kiểm thử đơn vị & hợp đồng dữ liệu (Unit & Contract Tests)
```bash
pytest tests/ -v
```

### Bước 5: Chạy benchmark phát triển trung gian (~10–15 phút)
Đánh giá trên 1.000 mẫu phân tầng để kiểm tra sơ bộ các baseline:
```bash
python scripts/attack_benchmark.py --config configs/development.yaml
```

### Bước 6: Chạy benchmark chính thức 10.000 mẫu bài báo (~2 giờ trên 2×T4)
Chạy toàn bộ 8 baseline và CASA qua 7 mốc ngân sách $K \in \{1, 2, 4, 8, 16, 32, 64\}$:
```bash
python scripts/attack_benchmark.py --config configs/paper_cifar10.yaml
```
Hoặc chạy riêng thuật toán CASA để thu thập số liệu chi tiết:
```bash
python scripts/run_casa_benchmark.py --config configs/paper_cifar10.yaml
```

### Bước 7: Chạy nghiên cứu phân rã thành phần (Ablation Study)
```bash
python scripts/run_ablation.py --config configs/ablation.yaml
```

### Bước 8: Chạy phân tích chẩn đoán thất bại (Failure Analysis)
```bash
python scripts/analyze_failures.py --results result/casa_10000_results.json
```

### Bước 9: Đánh giá thích ứng phòng thủ với BPDA
```bash
# Đánh giá bộ lọc Median 3x3 ở chế độ thích ứng BPDA
python scripts/defense_benchmark.py --config configs/development.yaml --defense median --mode adaptive
```

### Bước 10: Tự động sinh báo cáo và đồ thị xuất bản từ kết quả thô
```bash
# Sinh bảng biểu markdown
python scripts/generate_markdown_report.py \
    --results result/benchmark_results.json \
    --output result/benchmark_report.md

# Sinh hình vẽ 300 DPI sẵn sàng cho LaTeX
python scripts/plot_paper_figures.py \
    --results result/benchmark_results.json \
    --ablation result/ablation_results.json \
    --output-dir docs/assets/
```

---

## 3. Hướng Dẫn Thực Nghiệm Toàn Diện Trên Kaggle GPU (Kaggle Guide)

Kaggle cung cấp môi trường lý tưởng với **2× NVIDIA Tesla T4 (32 GB VRAM tổng cộng)** và 30 giờ GPU/tuần miễn phí. Dự án hỗ trợ sẵn sàng 1-click execution:

### 3.1 Khởi tạo Notebook trên Kaggle
1. Đăng nhập vào [Kaggle](https://www.kaggle.com/) và tạo một Notebook mới.
2. Tại bảng điều khiển bên phải (Notebook Options):
   - **Accelerator:** Chọn **GPU T4 x2**.
   - **Persistence:** Chọn **Files only** hoặc **Variables and Files**.
   - **Internet:** Bật **Always on**.

### 3.2 Tải mã nguồn & Checkpoint trên Kaggle Cell
Trong ô lệnh đầu tiên của Kaggle Notebook:
```python
# Clone repository
!git clone https://github.com/nxc1802/AA_2.git
%cd AA_2

# Cài đặt môi trường
!pip install -e . -q
```

### 3.3 Chạy qua Notebook chuyên dụng hoặc Master Shell Script
Dự án cung cấp sẵn tệp Notebook tự chứa [`kaggle_paper_runner.ipynb`](file:///Volumes/WorkSpace/Project/AA/kaggle_paper_runner.ipynb) và script tự động [`scripts/kaggle_run.sh`](file:///Volumes/WorkSpace/Project/AA/scripts/kaggle_run.sh):

```bash
# Thực thi toàn bộ pipeline tự động trên Kaggle terminal
bash scripts/kaggle_run.sh
```

Script này tự động:
1. Xác định thiết bị và kích hoạt song song 2 GPU (`DataParallel` hoặc đa tiến trình).
2. Tải và xác thực checkpoint SHA256.
3. Kích hoạt bộ nhớ đệm `attack_cache/` (nếu quá trình chạy bị gián đoạn hoặc hết thời gian quota, khi chạy lại sẽ tiếp tục từ mẫu gần nhất mà không cần tính lại từ đầu).
4. Đóng gói toàn bộ artifacts thành tệp nén: `cifar10_10k_artifacts.tar.gz`.

---

## 4. Tính Bất Biến & Kiểm Soát Ngẫu Nhiên (RNG Determinism)

Để đảm bảo kết quả tái lập tuyệt đối giữa các máy tính khác nhau:
1. **Sample-Bound Seed:** Mỗi mẫu ảnh $x_i$ được gán seed cục bộ:
   $$\text{Seed}_i = \text{Seed}_{\text{global}} + i \quad (i = 0, \dots, 9999)$$
   Điều này đảm bảo thứ tự thực thi đa luồng hoặc đa GPU không làm thay đổi chuỗi số ngẫu nhiên của từng mẫu.
2. **Deterministic cuDNN:** Khi kích hoạt cờ `--strict`, hệ thống tự động thiết lập:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
